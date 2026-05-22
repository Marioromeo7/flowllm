"""
Variant C (corrected) — GRU + shared notepad, supervised training.

The original Variant C had two flaws:
  1. Notepad only written from the last token position — 255/256 tokens never updated it
  2. Logged loss only at checkpoints (every 500 steps) — too sparse for analysis

Architecture per layer:
  all_h = nn.GRU(x)               cuDNN batched — all hidden states in one kernel call
  all_r = sigmoid(W_r * all_h)    batched gate (no sequential dependency)
  all_w = sigmoid(W_w * all_h)    batched gate (no sequential dependency)
  for t in 0..T:                  sequential only for notepad (unavoidable dependency)
    h_read_t = LN(h_t + r_t * note)
    note = (1-w_t)*note + w_t*h_t
  out = LN(MLP(all_h_reads) + all_h_reads)   batched MLP

GRU does not take h_read as input, so hidden states are notepad-independent.
Batching GRU + gates + MLP is mathematically identical to the GRUCell loop.
Notepad updates remain strictly sequential — causal write at t visible at t+1.

Loss: cross-entropy on next-token prediction (standard supervised).
Compare against Variant A (pure GRU) to isolate what the corrected notepad adds.
"""
import os
import csv
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
# CONFIG
# ==========================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN        = 256
D_MODEL        = 1024
N_LAYERS       = 6
BATCH_SIZE     = 4
TOTAL_STEPS    = 5000
SAVE_EVERY     = 500
LR             = 3e-4
SEED           = 42
CHECKPOINT_DIR = "checkpoints/variant_c_corrected"
LOG_PATH       = "checkpoints/variant_c_corrected/loss_log.csv"

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

def _prune(keep=1):
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
    # unwrap torch.compile if present
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Resumed from step {ckpt['step']} | loss {ckpt['loss']:.4f}", flush=True)
    return ckpt["step"]

# ==========================================
# MODEL
# ==========================================
class GRUNotepadBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru        = nn.GRU(d, d, batch_first=True)
        self.read_gate  = nn.Linear(d, d)
        self.write_gate = nn.Linear(d, d)
        self.mlp        = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln1        = nn.LayerNorm(d)
        self.ln2        = nn.LayerNorm(d)
        self.ln_skip    = nn.LayerNorm(d)

    def forward(self, x, note):
        B, T, d  = x.shape

        # Full sequence in one cuDNN call — h states are notepad-independent
        all_h, _ = self.gru(x)                                  # (B, T, d)

        # Batch gate projections — no sequential dependency on notepad
        all_r = torch.sigmoid(self.read_gate(all_h))            # (B, T, d)
        all_w = torch.sigmoid(self.write_gate(all_h))           # (B, T, d)

        # Sequential notepad accumulation — dependency chain, unavoidable
        note_vec = note.squeeze(1)                               # (B, d)
        h_reads  = []
        for t in range(T):
            h_read   = self.ln1(all_h[:, t] + all_r[:, t] * note_vec)
            note_vec = (1 - all_w[:, t]) * note_vec + all_w[:, t] * all_h[:, t]
            h_reads.append(h_read)

        h_reads = torch.stack(h_reads, dim=1)                   # (B, T, d)

        # Batched MLP — no sequential dependency
        out = self.ln2(self.mlp(h_reads) + h_reads)             # (B, T, d)

        # Block-level skip from input — gradient highway bypassing notepad chain
        out = self.ln_skip(out + x)                             # (B, T, d)

        return out, note_vec.unsqueeze(1)

class VariantC(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.layers  = nn.ModuleList([GRUNotepadBlock(d) for _ in range(N_LAYERS)])
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

    def forward(self, idx):
        B, T = idx.shape
        idx  = torch.clamp(idx, 0, self.tok_emb.num_embeddings - 1)
        x    = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device).clamp(0, SEQ_LEN - 1))
        note = torch.zeros(B, 1, D_MODEL, device=idx.device)
        for layer in self.layers:
            x, note = checkpoint(layer, x, note, use_reentrant=False)
        return self.lm_head(self.ln_f(x))

# ==========================================
# TRAINING
# ==========================================
def train():
    torch.cuda.empty_cache()

    model     = VariantC().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    p = sum(v.numel() for v in model.parameters())
    print(f"Variant C (corrected) — GRU + shared notepad | Params: {p:,} (~{p/1e6:.1f}M)", flush=True)

    start_step = load_latest(model, optimizer)
    if start_step == 0:
        init_log()

    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    dl = DataLoader(ds.shuffle(seed=SEED), batch_size=BATCH_SIZE,
                    collate_fn=collate_fn, num_workers=0)

    print(f"\n--- VARIANT C CORRECTED | steps {start_step} -> {TOTAL_STEPS} ---", flush=True)
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

        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
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

    print("\nVARIANT C CORRECTED COMPLETE.", flush=True)

if __name__ == "__main__":
    train()
