"""
Variant D — GRU + global notepad with cross-attention read.
Controlled variable: attention-based notepad read vs. gated read (Variant C).
All layers share ONE notepad (same as C), but instead of a scalar read gate,
each layer uses cross-attention to the notepad: Q from hidden state, K/V from notepad.
All hyperparameters identical to Variant C (flow_llm_300m.py).
"""
import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import AutoTokenizer
import time

# ==========================================
# CONFIG  (must match all other variants)
# ==========================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN        = 256
D_MODEL        = 1024
N_LAYERS       = 6
N_HEADS        = 8       # for notepad cross-attention, head_dim = 128
BATCH_SIZE     = 4
TOTAL_STEPS    = 5000
SAVE_EVERY     = 500
LR             = 3e-4
SEED           = 42
CHECKPOINT_DIR = "checkpoints/variant_d"
LOG_PATH       = "checkpoints/variant_d/loss_log.csv"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

def collate_fn(batch):
    toks = tokenizer([s["text"] for s in batch],
                     truncation=True, max_length=SEQ_LEN,
                     padding="max_length", return_tensors="pt")
    x = toks.input_ids[:, :-1]
    y = toks.input_ids[:, 1:]
    y[y == tokenizer.pad_token_id] = -100
    return x, y

# ==========================================
# CHECKPOINT HELPERS
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
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "loss": loss}, path)
    print(f"[ckpt] step {step} | loss {loss:.4f} -> {path}", flush=True)
    _prune(keep=1)

def _prune(keep=3):
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
                   key=lambda x: int(x.split("step")[1].split(".")[0]))
    for old in files[:-keep]:
        os.remove(os.path.join(CHECKPOINT_DIR, old))

def load_latest(model, optimizer):
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
                   key=lambda x: int(x.split("step")[1].split(".")[0]))
    if not files:
        print("[ckpt] Starting fresh.", flush=True)
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Resumed from step {ckpt['step']} | loss {ckpt['loss']:.4f}", flush=True)
    return ckpt["step"]

# ==========================================
# MODEL — GRU + shared notepad + attention read
# ==========================================
class NotepadCrossAttention(nn.Module):
    """
    Read from notepad via cross-attention.
    Q: from the layer's hidden state  (B, T, d)
    K, V: from the shared notepad     (B, 1, d)
    Output: attended notepad content  (B, T, d)

    Because the notepad has only 1 slot, the attention reduces to a
    learned, position-dependent scaling — strictly more expressive than
    the scalar read gate in Variant C.
    """
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d // n_heads
        self.scale    = self.head_dim ** -0.5
        self.q_proj   = nn.Linear(d, d)
        self.k_proj   = nn.Linear(d, d)
        self.v_proj   = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, notepad):
        # x: (B, T, d)   notepad: (B, 1, d)
        B, T, d = x.shape
        H, Hd   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)        # (B,H,T,Hd)
        k = self.k_proj(notepad).view(B, 1, H, Hd).transpose(1, 2)  # (B,H,1,Hd)
        v = self.v_proj(notepad).view(B, 1, H, Hd).transpose(1, 2)  # (B,H,1,Hd)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # (B,H,T,1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)          # (B,T,d)
        return self.out_proj(out)


class GRUNotepadAttnBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.gru        = nn.GRU(d, d, batch_first=True)
        self.mlp        = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )
        self.write_gate = nn.Linear(d, 1)
        self.notepad_attn = NotepadCrossAttention(d, n_heads)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)

    def forward(self, x, notepad, h_prev):
        if h_prev is None:
            h_prev = torch.zeros(1, x.size(0), x.size(2), device=x.device)

        gru_out, h_next = self.gru(x, h_prev)
        x_res = self.ln1(gru_out + x)

        # Write: update shared notepad with last-position summary
        w       = torch.sigmoid(self.write_gate(x_res))
        notepad = (1 - w[:, -1:, :]) * notepad + w[:, -1:, :] * x_res[:, -1:, :]

        # Read: cross-attend to notepad (richer than scalar gate)
        x_res = self.ln2(x_res + self.notepad_attn(x_res, notepad))

        out = self.ln3(self.mlp(x_res) + x_res)
        return out, notepad, h_next


class VariantD(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.layers  = nn.ModuleList(
            [GRUNotepadAttnBlock(d, N_HEADS) for _ in range(N_LAYERS)]
        )
        self.ln_f    = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, notepad=None, h_states=None):
        B, T = idx.shape
        idx  = torch.clamp(idx, 0, self.tok_emb.num_embeddings - 1)
        x    = (self.tok_emb(idx) +
                self.pos_emb(torch.arange(T, device=idx.device).clamp(0, SEQ_LEN - 1)))

        if notepad is None:
            notepad = torch.zeros(B, 1, self.tok_emb.embedding_dim, device=idx.device)
        if h_states is None:
            h_states = [None] * len(self.layers)

        new_h = []
        for i, layer in enumerate(self.layers):
            h_i = h_states[i]
            if h_i is None:
                h_i = torch.zeros(1, B, self.tok_emb.embedding_dim, device=idx.device)
            x, notepad, h_n = checkpoint(layer, x, notepad, h_i, use_reentrant=False)
            new_h.append(h_n.detach())

        return self.lm_head(self.ln_f(x)), notepad, new_h

# ==========================================
# TRAINING
# ==========================================
def train():
    torch.cuda.empty_cache()
    model     = VariantD().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    p = sum(v.numel() for v in model.parameters())
    print(f"Variant D — GRU + notepad + attention read | Params: {p:,} (~{p/1e6:.1f}M)", flush=True)

    start_step = load_latest(model, optimizer)
    if start_step == 0:
        init_log()

    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    dl = DataLoader(ds.shuffle(seed=SEED), batch_size=BATCH_SIZE,
                    collate_fn=collate_fn, num_workers=0)

    print(f"\n--- VARIANT D | steps {start_step} -> {TOTAL_STEPS} ---", flush=True)
    model.train()
    t0        = time.time()
    step      = start_step
    data_iter = iter(dl)

    while step < TOTAL_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)

        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits, _, _ = model(x)
        loss   = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 50 == 0:
            vram = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"Step {step:5d}/{TOTAL_STEPS} | Loss: {loss.item():.4f} | "
                  f"VRAM: {vram:.2f} GB | Time: {time.time()-t0:.1f}s", flush=True)
            log_loss(step, loss.item())
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, loss.item())

    if step % SAVE_EVERY != 0:
        save_checkpoint(model, optimizer, step, loss.item())

    print("\nVARIANT D COMPLETE.", flush=True)

if __name__ == "__main__":
    train()
