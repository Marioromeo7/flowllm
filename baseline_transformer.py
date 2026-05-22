import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import LambdaLR

# ==========================================
# 1. THE VOCABULARY & DATA
# ==========================================
CHARS = "0123456789+=\n"
VOCAB_SIZE = len(CHARS)

SEQ_LEN = 16 # Max sequence like "9+9=18\n" is only 6 chars, 16 is plenty safe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def encode(text):
    return [CHARS.index(c) for c in text]

def decode(ids):
    return "".join([CHARS[i] for i in ids if 0 <= i < VOCAB_SIZE])

def generate_math_batch(batch_size=32):
    inputs = []
    targets = []

    for _ in range(batch_size):
        # FIX 1: 1-digit addition. This is learnable in 30 seconds.
        # We will move to 3-digit once the baseline actually works.
        a = torch.randint(0, 9, (1,)).item()
        b = torch.randint(0, 9, (1,)).item()
        c = a + b

        text = f"{a}+{b}={c}\n"

        inp = encode(text[:-1])
        targ = encode(text[1:])

        inp += [0] * (SEQ_LEN - len(inp))
        targ += [-100] * (SEQ_LEN - len(targ))

        inputs.append(inp)
        targets.append(targ)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )

# ==========================================
# 2. CAUSAL SELF ATTENTION
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1) * -1e9
        att = att + mask
        att = torch.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

# ==========================================
# 3. TRANSFORMER BLOCK
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ==========================================
# 4. PURE TRANSFORMER
# ==========================================
class PureTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=3):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ==========================================
# 5. AUTOREGRESSIVE GENERATION (The brilliant part)
# ==========================================
@torch.no_grad()
def generate(model, prompt, max_new_tokens=8):
    model.eval()
    tokens = encode(prompt)

    for _ in range(max_new_tokens):
        if len(tokens) >= SEQ_LEN:
            break
        x = tokens + [0] * (SEQ_LEN - len(tokens))
        x = torch.tensor([x], dtype=torch.long, device=DEVICE)
        
        logits = model(x)[0]
        # Predict next token from the last REAL token
        next_token_logits = logits[len(tokens) - 1]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        tokens.append(next_token)
        if CHARS[next_token] == "\n":
            break # Stop when it predicts the newline

    return decode(tokens)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
def benchmark_baseline():
    print("INITIALIZING PURE TRANSFORMER BASELINE...")
    model = PureTransformer(VOCAB_SIZE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # Bumped LR slightly for 1-digit
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # FIX 2: 3000 steps. Takes ~30 seconds. Will NOT trigger Windows TDR crash.
    max_steps = 3000
    warmup_steps = 200

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()

        if step % 300 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            vram_used = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | LR: {current_lr:.5f} | VRAM: {vram_used:.2f} GB | Sample: {generate(model, '5+6=')}")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_time

    print("\n--- FINAL INFERENCE TEST ---")
    tests = ["2+3=", "7+8=", "5+6="]
    for t in tests:
        print(f"Input: {t:5} -> Predicted: {generate(model, t)}")

    print(f"\nTotal Train Time: {total_time:.2f} seconds")
    print("BASELINE METRICS CAPTURED. READY FOR HYBRID CHALLENGE.")

if __name__ == "__main__":
    benchmark_baseline()