"""
Variant H — Pure GRU, RL training with teacher log-probability reward.

Teacher: frozen gpt-neo-125M (well-trained, 125M params).
Student: pure GRU, 6 layers, d=1024.

Reward at each position = log probability gpt-neo assigns to the student's
sampled token given the context. Range (-inf, 0]: near 0 means the teacher
considered that token very likely; large negative means very unlikely.

Denser and smoother signal than cosine — the full teacher distribution informs
every reward, not just the single greedy token. No notepad.
Compare against Variant I to isolate notepad value.
"""
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# ==========================================
# CONFIG
# ==========================================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN        = 256
D_MODEL        = 1024
N_LAYERS       = 6
BATCH_SIZE     = 2
TOTAL_STEPS    = 5000
SAVE_EVERY     = 200
LR             = 3e-4
SEED           = 42
EMA_ALPHA      = 0.05
CHECKPOINT_DIR = "checkpoints/variant_h"
LOG_PATH       = "checkpoints/variant_h/loss_log.csv"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

def collate_fn(batch):
    toks = tokenizer([s["text"] for s in batch],
                     truncation=True, max_length=SEQ_LEN,
                     padding="max_length", return_tensors="pt")
    return toks.input_ids[:, :-1]

# ==========================================
# CHECKPOINT HELPERS
# ==========================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def init_log():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["step", "rl_loss", "mean_reward", "ce_loss"])

def log_loss(step, loss, mean_reward, ce_loss):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([step, loss, mean_reward, ce_loss])

def ce_eval(model, val_x, val_y):
    model.eval()
    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for i in range(val_x.size(0)):
            logits = model(val_x[i:i+1])
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), val_y[i:i+1].view(-1),
                                   ignore_index=tokenizer.pad_token_id, reduction='sum')
            n_tok += (val_y[i:i+1] != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item()
    model.train()
    return total_loss / max(n_tok, 1)

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
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Resumed from step {ckpt['step']} | loss {ckpt['loss']:.4f}", flush=True)
    return ckpt["step"]

# ==========================================
# STUDENT — Pure GRU
# ==========================================
class GRUBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), x.size(2), device=x.device)
        gru_out, _ = self.gru(x, h0)
        x = self.ln1(gru_out + x)
        x = self.ln2(self.mlp(x) + x)
        return x

class VariantH(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.pos_emb = nn.Embedding(SEQ_LEN, d)
        self.layers  = nn.ModuleList([GRUBlock(d) for _ in range(N_LAYERS)])
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
        idx = torch.clamp(idx, 0, self.tok_emb.num_embeddings - 1)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device).clamp(0, SEQ_LEN - 1))
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return self.lm_head(self.ln_f(x))

# ==========================================
# TRAINING
# ==========================================
def train():
    torch.cuda.empty_cache()

    print("Loading teacher (gpt-neo-125M)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", dtype=torch.float16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher loaded | Params: {t_params:,}", flush=True)

    student   = VariantH().to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=LR, foreach=False)
    s_params  = sum(p.numel() for p in student.parameters())
    print(f"Variant H — GRU Log-Prob RL | Params: {s_params:,} (~{s_params/1e6:.1f}M)", flush=True)

    start_step = load_latest(student, optimizer)
    if start_step == 0:
        init_log()

    print("Loading TinyStories...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    dl = DataLoader(ds.shuffle(seed=SEED), batch_size=BATCH_SIZE,
                    collate_fn=collate_fn, num_workers=0)

    # Fixed val batch for CE eval every 50 steps
    val_ds  = load_dataset("roneneldan/TinyStories", split="validation")
    val_raw = [val_ds[i] for i in range(4)]
    val_toks = tokenizer([s["text"] for s in val_raw], truncation=True,
                         max_length=SEQ_LEN, padding="max_length", return_tensors="pt")
    val_x = val_toks.input_ids[:, :-1].to(DEVICE)
    val_y = val_toks.input_ids[:, 1:].to(DEVICE)

    print(f"\n--- VARIANT H | steps {start_step} -> {TOTAL_STEPS} ---", flush=True)
    student.train()
    t0        = time.time()
    step      = start_step
    baseline  = 0.0
    data_iter = iter(dl)

    while step < TOTAL_STEPS:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x = next(data_iter)

        x = x.to(DEVICE)

        # Student forward — keep logits on GPU for backward
        s_logits = student(x)                                        # (B, T, V) fp32

        # Sample on CPU to avoid materialising 103 MB softmax on GPU
        with torch.no_grad():
            s_tokens = Categorical(logits=s_logits.detach().cpu()).sample().to(DEVICE)  # (B, T)

        # Log-probs via gather+logsumexp — no large intermediate allocated
        s_log_probs = (s_logits.gather(-1, s_tokens.unsqueeze(-1)).squeeze(-1)
                       - s_logits.logsumexp(dim=-1))                 # (B, T)

        # Teacher reward: move teacher to GPU only for its forward pass
        teacher.cuda()
        with torch.no_grad():
            t_logits = teacher(x).logits                             # (B, T, V) fp16
            reward   = (t_logits.gather(-1, s_tokens.unsqueeze(-1)).squeeze(-1).float()
                        - t_logits.logsumexp(dim=-1).float())        # (B, T) fp32
            del t_logits
        teacher.cpu()
        torch.cuda.empty_cache()

        pad_mask    = (x != tokenizer.pad_token_id).float()
        reward      = reward * pad_mask
        n_valid     = pad_mask.sum().item()
        mean_reward = reward.sum().item() / max(n_valid, 1)
        baseline    = (1 - EMA_ALPHA) * baseline + EMA_ALPHA * mean_reward
        advantage   = (reward - baseline) * pad_mask
        loss        = -(advantage * s_log_probs).sum() / max(n_valid, 1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        loss_val = loss.item()
        del x, s_logits, s_tokens, s_log_probs, reward, pad_mask, advantage, loss
        torch.cuda.empty_cache()
        optimizer.step()
        step += 1

        if step % 50 == 0:
            torch.cuda.empty_cache()
            vram = torch.cuda.max_memory_allocated(0) / 1024**3
            ce   = ce_eval(student, val_x, val_y)
            print(f"Step {step:5d}/{TOTAL_STEPS} | Loss: {loss_val:.4f} | "
                  f"Reward: {mean_reward:.4f} | Baseline: {baseline:.4f} | "
                  f"CE: {ce:.4f} | VRAM: {vram:.2f} GB | Time: {time.time()-t0:.1f}s", flush=True)
            log_loss(step, loss_val, mean_reward, ce)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        if step % SAVE_EVERY == 0:
            save_checkpoint(student, optimizer, step, loss_val)

    if step % SAVE_EVERY != 0:
        save_checkpoint(student, optimizer, step, loss_val)

    print("\nVARIANT H COMPLETE.", flush=True)

if __name__ == "__main__":
    train()
