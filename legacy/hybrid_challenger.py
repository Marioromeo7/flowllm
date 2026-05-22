import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import LambdaLR

# ==========================================
# 1. DATA & VOCAB
# ==========================================
CHARS = "0123456789+=\n"
VOCAB_SIZE = len(CHARS)
SEQ_LEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def encode(text): return [CHARS.index(c) for c in text]
def decode(ids): return "".join([CHARS[i] for i in ids if 0 <= i < VOCAB_SIZE])

def generate_math_batch(batch_size=32):
    inputs, targets = [], []
    for _ in range(batch_size):
        a = torch.randint(0, 9, (1,)).item()
        b = torch.randint(0, 9, (1,)).item()
        text = f"{a}+{b}={a+b}\n"
        inp = encode(text[:-1]) + [0] * (SEQ_LEN - len(encode(text[:-1])))
        targ = encode(text[1:]) + [-100] * (SEQ_LEN - len(encode(text[1:])))
        inputs.append(inp)
        targets.append(targ)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# ==========================================
# 2. PATH B: SOFT ENTROPY ROUTER
# ==========================================
class FlowEngine(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            return self.gru(x)[0] 

class CacheEngine(nn.Module):
    def __init__(self, d_model, n_heads, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        x_window = x[:, -self.window_size:, :] 
        W = x_window.shape[1]
        qkv = self.qkv(x_window)
        q, k, v = qkv.split(self.head_dim * self.n_heads, dim=2)
        q = q.view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(W, W, device=x.device), diagonal=1) * -1e9
        att = att + mask
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, W, C)
        out = self.proj(y)
        if T > W:
            out = torch.cat([torch.zeros(B, T - W, C, device=x.device), out], dim=1)
        return out

class SoftEntropyRouter(nn.Module):
    """Outputs a continuous value. High = Flow, Low = Cache."""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # Output raw logits
        raw_gate = self.gate(x)
        
        # THE BIAS FIX: Add a large negative bias (-2.0) before sigmoid.
        # This means untrained, the sigmoid outputs ~0.12, heavily favoring Cache (stable).
        # The model has to LEARN to push the bias high to use Flow.
        biased_gate = raw_gate - 2.0 
        return torch.sigmoid(biased_gate) 

class HybridModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        
        self.flow = FlowEngine(d_model)
        self.cache = CacheEngine(d_model, n_heads, window_size=8)
        self.router = SoftEntropyRouter(d_model) # New Router
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        
        flow_out = self.flow(x)
        cache_out = self.cache(x)
        
        # Soft continuous mixing. Smooth gradients!
        gate = self.router(x) # Shape: (Batch, Seq, 1)
        x = (gate * flow_out) + ((1 - gate) * cache_out)
        
        x = self.ln_f(x)
        return self.lm_head(x)

# ==========================================
# 3. INFERENCE & TRAINING
# ==========================================
@torch.no_grad()
def generate(model, prompt, max_new_tokens=8):
    model.eval()
    tokens = encode(prompt)
    for _ in range(max_new_tokens):
        if len(tokens) >= SEQ_LEN: break
        x = torch.tensor([tokens + [0] * (SEQ_LEN - len(tokens))], dtype=torch.long, device=DEVICE)
        logits = model(x)[0]
        next_token = torch.argmax(logits[len(tokens) - 1], dim=-1).item()
        tokens.append(next_token)
        if CHARS[next_token] == "\n": break
    return decode(tokens)

def benchmark_hybrid():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("INITIALIZING PATH B: SOFT ENTROPY ROUTER...")
    model = HybridModel(VOCAB_SIZE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    max_steps = 3000
    warmup_steps = 200
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"\n--- STARTING TRAINING ({max_steps} Steps) ---")
    start_time = time.time()
    model.train()

    for step in range(max_steps):
        inputs, targets = generate_math_batch(batch_size=64)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        logits = model(inputs)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 300 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            vram_used = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | LR: {current_lr:.5f} | VRAM: {vram_used:.2f} GB | Sample: {generate(model, '5+6=')}")
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_time

    print("\n--- FINAL INFERENCE TEST ---")
    for t in ["2+3=", "7+8=", "5+6="]:
        print(f"Input: {t:5} -> Predicted: {generate(model, t)}")

    print(f"\nTotal Train Time: {total_time:.2f} seconds")
    print("PATH B METRICS CAPTURED.")

if __name__ == "__main__":
    benchmark_hybrid()