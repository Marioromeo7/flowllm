import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint
import time

# ==========================================
# 1. SETUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 256

CHECKPOINT_DIR = "checkpoints/variant_c"
LOG_PATH       = "checkpoints/variant_c/loss_log.csv"
SAVE_EVERY     = 500
TOTAL_STEPS    = 5000

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

def collate_fn(batch):
    texts = [story['text'] for story in batch]
    tokens = tokenizer(texts, truncation=True, max_length=SEQ_LEN, padding="max_length", return_tensors="pt")
    x = tokens.input_ids[:, :-1]
    y = tokens.input_ids[:, 1:]
    y[y == tokenizer.pad_token_id] = -100
    return x, y

# ==========================================
# 2. CHECKPOINT HELPERS
# ==========================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def init_log():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss"])

def log_loss(step, loss):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([step, loss])

def save_checkpoint(model, optimizer, step, loss):
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_step{step}.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss":      loss,
    }, path)
    print(f"[ckpt] step {step} | loss {loss:.4f} -> {path}")
    _prune_checkpoints(keep=1)

def _prune_checkpoints(keep=1):
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("step")[1].split(".")[0])
    )
    for old in files[:-keep]:
        os.remove(os.path.join(CHECKPOINT_DIR, old))

def load_latest_checkpoint(model, optimizer):
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("step")[1].split(".")[0])
    )
    if not files:
        print("[ckpt] No checkpoint found. Starting fresh.")
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Resumed from step {ckpt['step']} | loss {ckpt['loss']:.4f}")
    return ckpt["step"]

# ==========================================
# 3. THE 140M FLOW LLM
# ==========================================
class FlowLLMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.write_gate = nn.Linear(d_model, 1)
        self.read_gate  = nn.Linear(d_model, 1)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, notepad, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(1, x.size(0), x.size(2), device=x.device)
        gru_out, h_next = self.gru(x, h_prev)
        x_res = self.ln1(gru_out + x)

        w = torch.sigmoid(self.write_gate(x_res))
        notepad = (1 - w[:, -1:, :]) * notepad + w[:, -1:, :] * x_res[:, -1:, :]
        r = torch.sigmoid(self.read_gate(x_res))
        x_res = x_res + (r * notepad)

        out = self.ln2(self.mlp(x_res) + x_res)
        return out, notepad, h_next


class FlowLLM(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layers=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.layers  = nn.ModuleList([FlowLLMBlock(d_model) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, notepad=None, h_states=None):
        B, T = idx.shape
        idx = torch.clamp(idx, 0, self.tok_emb.num_embeddings - 1)

        positions = torch.arange(T, device=idx.device).clamp(0, SEQ_LEN - 1)
        x = self.tok_emb(idx) + self.pos_emb(positions)

        if notepad is None:
            notepad = torch.zeros(B, 1, self.tok_emb.embedding_dim, device=idx.device)
        if h_states is None:
            h_states = [None] * len(self.layers)

        new_h_states = []
        for i, layer in enumerate(self.layers):
            h_i = h_states[i]
            if h_i is None:
                h_i = torch.zeros(1, B, self.tok_emb.embedding_dim, device=idx.device)
            x, notepad, h_n = checkpoint(layer, x, notepad, h_i, use_reentrant=False)
            new_h_states.append(h_n.detach())  # detach: don't backprop across windows

        return self.lm_head(self.ln_f(x)), notepad, new_h_states

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_flow_llm():
    torch.cuda.empty_cache()
    model = FlowLLM(VOCAB_SIZE, d_model=1024, n_layers=6).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    start_step = load_latest_checkpoint(model, optimizer)
    if start_step == 0:
        init_log()

    print("Loading TinyStories dataset...")
    dataset    = load_dataset("roneneldan/TinyStories", split="train")
    dataloader = DataLoader(dataset.shuffle(seed=42), batch_size=4,
                            collate_fn=collate_fn, num_workers=0)

    print(f"\n--- VARIANT C: GRU + SHARED NOTEPAD | steps {start_step} -> {TOTAL_STEPS} ---")
    model.train()
    start_time = time.time()
    step = start_step
    data_iter = iter(dataloader)

    while step < TOTAL_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, _, _ = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % 50 == 0:
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f"Step {step:5d}/{TOTAL_STEPS} | Loss: {loss.item():.4f} | VRAM: {vram:.2f} GB | Time: {elapsed:.1f}s")
            log_loss(step, loss.item())
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, loss.item())

    # Final checkpoint if not already saved
    if step % SAVE_EVERY != 0:
        save_checkpoint(model, optimizer, step, loss.item())

    # ==========================================
    # 5. GENERATION
    # ==========================================
    print("\n--- GENERATING STORY ---")
    prompt = "Once upon a time, little"
    inputs_gen = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    model.eval()
    notepad  = None
    h_states = None
    generated = inputs_gen.clone()

    with torch.no_grad():
        for _ in range(100):
            window = generated[:, -SEQ_LEN:]
            logits, notepad, h_states = model(window, notepad, h_states)
            probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).reshape(1, 1)
            generated = torch.cat([generated, next_token], dim=1)

    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    print("\nTRAINING COMPLETE.")

if __name__ == "__main__":
    train_flow_llm()
