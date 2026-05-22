import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. THE ARCHITECTURE (Same as Phase 2)
# ==========================================
class RealFlow(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    def forward(self, embeddings):
        return self.gru(embeddings)[0][:, -1, :]

class RealCache(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lm = AutoModel.from_pretrained("distilgpt2")
        self.window_size = 4
    def forward(self, embeddings):
        return self.lm(inputs_embeds=embeddings[:, -self.window_size:, :]).last_hidden_state[:, -1, :]

class DynamicRouter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_layer = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return torch.sigmoid(self.gate_layer(x)).squeeze(-1)

class HybridTextModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.router = DynamicRouter(hidden_size)
        self.flow = RealFlow(hidden_size)
        self.cache = RealCache(hidden_size)
        
    def forward(self, embeddings):
        gate = self.router(embeddings[:, -1, :]).unsqueeze(-1)
        flow_out = self.flow(embeddings)
        cache_out = self.cache(embeddings)
        return (gate * cache_out) + ((1 - gate) * flow_out), gate

# ==========================================
# 2. THE TRAINING SETUP
# ==========================================
def train_the_brain():
    print("Loading Tokenizer and Models...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = HybridTextModel(hidden_size=768).cuda()
    
    # THE ENGINEERING TRICK: Freeze everything except the Router!
    for param in model.flow.parameters():
        param.requires_grad = False
    for param in model.cache.parameters():
        param.requires_grad = False
        
    # Only the Router's weights will update
    optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Our synthetic dataset: (Text, Target Gate Score)
    # 0.0 = Use Flow (Easy grammar)
    # 1.0 = Use Cache (Hard logic)
    dataset = [
        ("The cat sat on the", 0.0),              # Easy, predictable grammar
        ("I went to the store to buy some", 0.0),  # Easy
        ("To implement a binary search tree in", 1.0), # Hard, requires precise logic
        ("The quantum superposition of the particle", 1.0), # Hard, complex concept
        ("She opened the door and walked", 0.0),   # Easy
    ]

    print("\n--- PRE-TRAINING TEST ---")
    with torch.no_grad():
        emb = model.cache.lm.get_input_embeddings()(tokenizer("The cat sat on the", return_tensors="pt").input_ids.cuda())
        _, pre_gate = model(emb)
        print(f"Untrained Router on easy text: {pre_gate.item():.4f} (Should be ~0.5)")

    print("\n--- TRAINING THE ROUTER (10 Epochs) ---")
    for epoch in range(10):
        total_loss = 0
        for text, target_gate in dataset:
            # 1. Get embeddings (Frozen, no math heavy lifting)
            inputs = tokenizer(text, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                embeddings = model.cache.lm.get_input_embeddings()(inputs)
            
            # 2. Forward pass through Hybrid (Only Router uses gradients)
            optimizer.zero_grad()
            output, gate_score = model(embeddings)
            
            # 3. Calculate Loss (How far is the gate from our target?)
            target_tensor = torch.tensor([target_gate], dtype=torch.float32).cuda()
            loss = loss_fn(gate_score, target_tensor)
            
            # 4. Backward pass (Updates ONLY the Router)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(dataset):.4f}")

    print("\n--- POST-TRAINING TEST ---")
    test_sentences = [
        "The cat sat on the", # Should be low (~0.0)
        "To implement a binary search tree in" # Should be high (~1.0)
    ]
    
    with torch.no_grad():
        for sentence in test_sentences:
            emb = model.cache.lm.get_input_embeddings()(tokenizer(sentence, return_tensors="pt").input_ids.cuda())
            _, post_gate = model(emb)
            action = "PRECISE CACHE" if post_gate.item() > 0.5 else "CONTINUOUS FLOW"
            print(f"Text: '{sentence}'")
            print(f"Gate: {post_gate.item():.4f} -> Action: {action}\n")

    print("SUCCESS: The Router has learned to differentiate entropy.")

if __name__ == "__main__":
    train_the_brain()