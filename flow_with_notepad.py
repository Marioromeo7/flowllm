import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import LambdaLR

# ==========================================
# 1. DATA & VOCAB (3-Digit Math Immediately)
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
        a = torch.randint(0, 999, (1,)).item()
        b = torch.randint(0, 999, (1,)).item()
        text = f"{a}+{b}={a+b}\n"
        inp = encode(text[:-1]) + [0] * (SEQ_LEN - len(encode(text[:-1])))
        targ = encode(text[1:]) + [-100] * (SEQ_LEN - len(encode(text[1:])))
        inputs.append(inp)
        targets.append(targ)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# ==========================================
# 2. THE FLOW ENGINE WITH FIXED MICRO-CACHE
# ==========================================
class FlowNotepadModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # Use GRUCell so we can manually step through and update the notepad
        self.gru_cell = nn.GRUCell(d_model, d_model)
        
        # THE MICRO-CACHE LOGIC
        # Write Gate: "Is this important? (e.g., a carry)"
        self.write_gate = nn.Linear(d_model, 1)
        # Read Gate: "Do I need the notepad right now?"
        self.read_gate = nn.Linear(d_model, 1)
        
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
        x = self.tok_emb(idx)
        
        # Initialize the Liquid State (Flow)
        h = torch.zeros(B, self.gru_cell.hidden_size, device=idx.device)
        
        # Initialize the Fixed Micro-Cache (The Notepad)
        # This is the only "memory" it has outside the GRU.
        # Size: (Batch, Hidden Size) -> ~1KB of VRAM. Fixed. Never grows.
        notepad = torch.zeros(B, self.gru_cell.hidden_size, device=idx.device)
        
        outputs = []
        
        # Step through the sequence character by character
        for t in range(T):
            x_t = x[:, t, :]
            
            # 1. Flow Update (Continuous State)
            h = self.gru_cell(x_t, h)
            
            # 2. Write to Notepad
            # sigmoid decides if it should keep old notepad or overwrite with new 'h'
            w = torch.sigmoid(self.write_gate(h))
            notepad = (1 - w) * notepad + w * h
            
            # 3. Read from Notepad
            # sigmoid decides how much of the notepad to add to the current thought
            r = torch.sigmoid(self.read_gate(h))
            final_h = h + (r * notepad)
            
            outputs.append(final_h)
            
        # Stack outputs back into sequence format
        x = torch.stack(outputs, dim=1)
        x = self.ln_f(x)
        return self.lm_head(x)

# ==========================================
# 3. INFERENCE & TRAINING
# ==========================================
@torch.no_grad()
def generate(model, prompt, max_new_tokens=10):
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

def train_notepad():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("INITIALIZING FLOW + FIXED MICRO-CACHE...")
    model = FlowNotepadModel(VOCAB_SIZE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    max_steps = 3000
    warmup = 200
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = float(s - warmup) / float(max(1, max_steps - warmup))
        return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"\n--- TRAINING ON 3-DIGIT MATH (999+999) ---")
    start_time = time.time()
    model.train()

    for step in range(max_steps):
        inputs, targets = generate_math_batch()
        logits = model(inputs.to(DEVICE))
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.to(DEVICE).view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 300 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            vram = torch.cuda.max_memory_allocated(0) / (1024**3)
            # The real test: Did it learn the carries?
            sample = generate(model, "999+999=")
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | VRAM: {vram:.2f} GB | 999+999={sample.split('=')[1].strip()}")
            torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_time
    print("\n--- FINAL INFERENCE TEST ---")
    tests = ["23+45=", "57+88=", "123+456=", "999+999="]
    for t in tests:
        pred = generate(model, t)
        print(f"Input: {t:10} -> Pred: {pred.strip()}")

    print(f"\nTrain Time: {total_time:.2f} sec")
    print("MICRO-CACHE METRICS CAPTURED.")

if __name__ == "__main__":
    train_notepad()