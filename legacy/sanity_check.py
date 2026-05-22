import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

DEVICE   = "cuda"
SEQ_LEN  = 256
D_MODEL  = 1024
N_LAYERS = 6
STEPS    = 10

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB total", flush=True)

def collate_fn(batch):
    toks = tokenizer([s["text"] for s in batch],
                     truncation=True, max_length=SEQ_LEN,
                     padding="max_length", return_tensors="pt")
    x = toks.input_ids[:, :-1]
    y = toks.input_ids[:, 1:]
    y[y == tokenizer.pad_token_id] = -100
    return x, y

class FlowLLMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.wg  = nn.Linear(d, 1)
        self.rg  = nn.Linear(d, 1)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, notepad, h_prev):
        gru_out, h_next = self.gru(x, h_prev)
        r = self.ln1(gru_out + x)
        w = torch.sigmoid(self.wg(r))
        notepad = (1 - w[:, -1:, :]) * notepad + w[:, -1:, :] * r[:, -1:, :]
        r = r + torch.sigmoid(self.rg(r)) * notepad
        return self.ln2(self.mlp(r) + r), notepad, h_next

class FlowLLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.emb    = nn.Embedding(VOCAB_SIZE, d)
        self.pos    = nn.Embedding(SEQ_LEN, d)
        self.layers = nn.ModuleList([FlowLLMBlock(d) for _ in range(N_LAYERS)])
        self.ln     = nn.LayerNorm(d)
        self.head   = nn.Linear(d, VOCAB_SIZE, bias=False)
        self.head.weight = self.emb.weight
        self.d = d

    def forward(self, idx):
        B, T = idx.shape
        idx = torch.clamp(idx, 0, self.emb.num_embeddings - 1)
        x   = self.emb(idx) + self.pos(torch.arange(T, device=idx.device).clamp(0, SEQ_LEN - 1))
        pad = torch.zeros(B, 1, self.d, device=idx.device)
        for layer in self.layers:
            h0 = torch.zeros(1, B, self.d, device=idx.device)
            x, pad, _ = checkpoint(layer, x, pad, h0, use_reentrant=False)
        return self.head(self.ln(x))

torch.cuda.empty_cache()
model = FlowLLM().to(DEVICE)
p = sum(v.numel() for v in model.parameters())
print(f"Params: {p:,} (~{p/1e6:.1f}M)", flush=True)

opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

print("Loading dataset...", flush=True)
ds = load_dataset("roneneldan/TinyStories", split="train")
dl = iter(DataLoader(ds.shuffle(seed=42).select(range(60)), batch_size=4,
                     collate_fn=collate_fn, num_workers=0))

print(f"--- {STEPS}-step sanity check (Variant C: GRU + shared notepad) ---", flush=True)
t0 = time.time()
model.train()

for step in range(STEPS):
    x, y = next(dl)
    x, y = x.to(DEVICE), y.to(DEVICE)

    logits = model(x)
    loss   = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    vram = torch.cuda.max_memory_allocated(0) / 1024**3
    dt   = time.time() - t0
    print(f"  step {step:2d} | loss {loss.item():.4f} | VRAM {vram:.2f} GB | {dt:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()

print("SANITY CHECK PASSED — ready for Phase 1 run.", flush=True)
sys.exit(0)
