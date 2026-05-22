import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. THE REAL ENGINES
# ==========================================
class RealFlow(nn.Module):
    """Continuous State Model (GRU). No cache, just a liquid hidden state."""
    def __init__(self, hidden_size):
        super().__init__()
        # GRU maintains a continuous state that evolves over time
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, embeddings):
        # We don't need past tokens! Just the current sequence and the evolving state.
        # output shape: (Batch, SeqLen, Hidden)
        output, _ = self.gru(embeddings)
        # We only care about the very last step's state
        return output[:, -1, :] 

class RealCache(nn.Module):
    """Discrete Sliding Window Cache (DistilGPT2)."""
    def __init__(self, hidden_size):
        super().__init__()
        # Load a tiny real language model to act as our precise cache
        self.lm = AutoModel.from_pretrained("distilgpt2")
        self.window_size = 4 # Only look at the last 4 tokens!
        
    def forward(self, embeddings):
        # THE SLIDING WINDOW: Chop off everything except the last 4 tokens
        windowed_input = embeddings[:, -self.window_size:, :]
        # Process through the real Transformer
        lm_out = self.lm(inputs_embeds=windowed_input)
        # Get the precise output of the last token
        return lm_out.last_hidden_state[:, -1, :]

# ==========================================
# 2. YOUR INVENTION: THE DYNAMIC ROUTER
# ==========================================
class DynamicRouter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.gate_layer(x)).squeeze(-1)

# ==========================================
# 3. THE HYBRID ARCHITECTURE
# ==========================================
class HybridTextModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.router = DynamicRouter(hidden_size)
        self.flow = RealFlow(hidden_size)
        self.cache = RealCache(hidden_size)
        
    def forward(self, embeddings):
        # 1. Route based on the raw embeddings
        gate = self.router(embeddings[:, -1, :]) # Look at current token to decide
        gate = gate.unsqueeze(-1)
        
        # 2. Get outputs from both engines
        flow_out = self.flow(embeddings)
        cache_out = self.cache(embeddings)
        
        # 3. Mix
        final_output = (gate * cache_out) + ((1 - gate) * flow_out)
        return final_output, gate

# ==========================================
# 4. THE REAL TEXT TEST
# ==========================================
def test_real_text():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    print("Initializing Real Hybrid Model...")
    # DistilGPT2 has a hidden size of 768
    model = HybridTextModel(hidden_size=768)
    model.cuda()
    
    # A sentence with both "easy" grammar and a "hard" logic setup
    text = "The cat sat on the mat because it was"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get word embeddings (the raw numbers representing the words)
    with torch.no_grad():
        embeddings = model.cache.lm.get_input_embeddings()(inputs.input_ids.cuda())
    
    print(f"\nInput Text: '{text}'")
    print("Processing through Hybrid Engines...")
    
    # Run the hybrid model
    output, gate_score = model(embeddings)
    
    print(f"\n--- ROUTING DECISION ---")
    print(f"Gate Score: {gate_score.item():.4f}")
    
    if gate_score.item() > 0.5:
        print("Action: Routed to PRECISE CACHE (Transformer)")
    else:
        print("Action: Routed to CONTINUOUS FLOW (GRU)")
        
    print("\nSUCCESS: Real text processed. Phase 2 complete.")

if __name__ == "__main__":
    test_real_text()