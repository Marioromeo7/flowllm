import torch
import torch.nn as nn

# ==========================================
# 1. THE FROZEN ENGINES (Fake for now)
# ==========================================
class FakeFlow(nn.Module):
    """Represents your Continuous State/Flow model (e.g., Mamba)."""
    def __init__(self, hidden_size):
        super().__init__()
        # A simple linear layer to simulate flow processing
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)

class FakeCache(nn.Module):
    """Represents your Discrete Sliding Window Cache (e.g., Llama)."""
    def __init__(self, hidden_size):
        super().__init__()
        # A simple linear layer to simulate precise cache retrieval
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)

# ==========================================
# 2. YOUR INVENTION: THE DYNAMIC ROUTER
# ==========================================
class DynamicRouter(nn.Module):
    """
    This is the brain of the operation. 
    It looks at a token and decides: Flow (0.0) or Cache (1.0)?
    """
    def __init__(self, hidden_size):
        super().__init__()
        # A tiny layer to calculate the "entropy/gate" score
        self.gate_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Sigmoid forces the output to be exactly between 0.0 and 1.0
        # Shape: (Batch, 1) -> we squeeze it to (Batch,)
        gate_score = torch.sigmoid(self.gate_layer(x)).squeeze(-1)
        return gate_score

# ==========================================
# 3. THE HYBRID ARCHITECTURE
# ==========================================
class Hybrid3050Killer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.flow = FakeFlow(hidden_size)
        self.cache = FakeCache(hidden_size)
        self.router = DynamicRouter(hidden_size)
        
    def forward(self, x):
        # Step 1: Ask the router what to do
        # gate shape: [Batch_Size] (e.g., [0.2, 0.9, 0.1...])
        gate = self.router(x)
        
        # Step 2: Process through BOTH engines
        flow_out = self.flow(x)
        cache_out = self.cache(x)
        
        # Step 3: The Mixing Trick
        # We need to reshape gate so PyTorch can multiply it with the outputs
        # gate becomes shape: [Batch_Size, 1]
        gate = gate.unsqueeze(-1)
        
        # If gate is 0.8 -> 80% Cache, 20% Flow
        # If gate is 0.1 -> 10% Cache, 90% Flow
        final_output = (gate * cache_out) + ((1 - gate) * flow_out)
        
        return final_output, gate

# ==========================================
# 4. THE AUTOGRAD STRESS TEST
# ==========================================
def test_architecture():
    print("Initializing Hybrid Architecture...")
    hidden_size = 64
    model = Hybrid3050Killer(hidden_size)
    
    # Move everything to the 3050
    model.cuda()
    
    print("Creating dummy token data (Batch of 8 tokens)...")
    # Simulating 8 tokens, each with 64 features
    dummy_tokens = torch.randn(8, hidden_size).cuda()
    
    print("Running Forward Pass...")
    output, gate_scores = model(dummy_tokens)
    
    print(f"Output shape: {output.shape}") # Should be [8, 64]
    print(f"Router Gate Scores: {gate_scores.detach().cpu().numpy()}") 
    # Look closely at these numbers! Are they close to 0 or 1?
    
    print("\nRunning Backward Pass (The Calculus Test)...")
    # We create a fake "error" to force the calculus engine to wake up
    fake_target = torch.randn_like(output)
    loss = nn.MSELoss()(output, fake_target)
    
    # THIS IS THE MAGIC LINE. Does the math flow backwards through the gate?
    loss.backward()
    
    # Check if the Router actually learned (calculated gradients)
    router_grad = model.router.gate_layer.weight.grad
    
    if router_grad is None:
        print("\nFATAL ERROR: Autograd died. The calculus didn't flow through the Router.")
        return False
        
    if torch.isnan(router_grad).any():
        print("\nFATAL ERROR: NaN detected. The math exploded.")
        return False
        
    print(f"Router gradients calculated successfully! Mean gradient: {router_grad.mean().item():.6f}")
    print("\nSUCCESS: The Hybrid Skeleton is structurally sound. Phase 1 complete.")
    return True

if __name__ == "__main__":
    test_architecture()