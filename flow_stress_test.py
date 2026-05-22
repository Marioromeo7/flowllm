import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import LambdaLR

CHARS = "0123456789+=\n"
VOCAB_SIZE = len(CHARS)
SEQ_LEN = 16 # Big enough for "999+999=1998\n"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def encode(text): return [CHARS.index(c) for c in text]
def decode(ids): return "".join([CHARS[i] for i in ids if 0 <= i < VOCAB_SIZE])

def generate_math_batch(batch_size=32, max_num=9):
    inputs, targets = [], []
    for _ in range(batch_size):
        a = torch.randint(0, max_num, (1,)).item()
        b = torch.randint(0, max_num, (1,)).item()
        text = f"{a}+{b}={a+b}\n"
        inp = encode(text[:-1]) + [0] * (SEQ_LEN - len(encode(text[:-1])))
        targ = encode(text[1:]) + [-100] * (SEQ_LEN - len(encode(text[1:])))
        inputs.append(inp)
        targets.append(targ)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

class PureFlowModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
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
        x = self.tok_emb(idx)
        with torch.backends.cudnn.flags(enabled=False):
            gru_out, _ = self.gru(x)
        x = self.ln_f(gru_out)
        return self.lm_head(x)

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

def stress_test():
    torch.cuda.empty_cache()
    model = PureFlowModel(VOCAB_SIZE).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    max_steps = 3000
    warmup = 200
    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = float(s - warmup) / float(max(1, max_steps - warmup))
        return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Train it purely on 1-digit math to establish a baseline
    print("\n--- TRAINING ON 1-DIGIT (0-9) ---")
    model.train()
    for step in range(max_steps):
        inputs, targets = generate_math_batch(max_num=9)
        logits = model(inputs.to(DEVICE))
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.to(DEVICE).view(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

    # NOW STRESS TEST IT ON INCREASINGLY HARDER MATH
    print("\n--- STRESS TEST: CAN THE FLOW GENERALIZE? ---")
    tests = [
        ("1-Digit", "5+6="),
        ("2-Digit", "23+45="),
        ("2-Digit Carry", "57+88="),
        ("3-Digit", "123+456="),
        ("3-Digit Carry", "999+999=")
    ]
    
    for name, prompt in tests:
        pred = generate(model, prompt)
        correct = prompt.split("=")[0] in pred and pred.endswith("\n")
        status = "✅ PASS" if correct else "❌ FAIL"
        print(f"{status} | {name:15} | Input: {prompt:10} -> Pred: {pred.strip()}")

    print("\n--- RE-TRAINING ON 3-DIGIT (0-999) ---")
    # Reset optimizer for new task
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    model.train()
    for step in range(max_steps):
        inputs, targets = generate_math_batch(max_num=999) # Hard math
        logits = model(inputs.to(DEVICE))
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), targets.to(DEVICE).view(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        if step % 1000 == 0: print(f"Step {step} Loss: {loss.item():.4f}")

    print("\n--- STRESS TEST AFTER 3-DIGIT TRAINING ---")
    for name, prompt in tests:
        pred = generate(model, prompt)
        correct = prompt.split("=")[0] in pred and pred.endswith("\n")
        status = "✅ PASS" if correct else "❌ FAIL"
        print(f"{status} | {name:15} | Input: {prompt:10} -> Pred: {pred.strip()}")

if __name__ == "__main__":
    stress_test()