import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
D_MODEL = 1024
N_LAYERS = 6
SEQ_LEN = 256

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = tokenizer.vocab_size

print(f"Free VRAM before teacher: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB", flush=True)

teacher = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", dtype=torch.float16).to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print(f"Teacher loaded. Free VRAM: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB", flush=True)

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
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device).clamp(0, SEQ_LEN-1))
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return self.lm_head(self.ln_f(x))

print("Creating student...", flush=True)
student = VariantH().to(DEVICE)
print(f"Student loaded. Free VRAM: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB", flush=True)

print("Creating optimizer...", flush=True)
import torch.optim as optim
optimizer = optim.AdamW(student.parameters(), lr=3e-4)
print("Optimizer created.", flush=True)

print("Running one forward pass...", flush=True)
x = torch.randint(0, VOCAB_SIZE, (4, 255), device=DEVICE)
with torch.no_grad():
    t_out = teacher(x).logits
print(f"Teacher forward ok. Shape: {t_out.shape}", flush=True)

s_out = student(x)
print(f"Student forward ok. Shape: {s_out.shape}", flush=True)
print("ALL OK", flush=True)
