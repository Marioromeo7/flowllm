import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. THE ENGINES & ROUTER
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

# ==========================================
# 2. THE HARD SWITCH ARCHITECTURE
# ==========================================
class HardSwitchHybrid(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.router = DynamicRouter(hidden_size)
        self.flow = RealFlow(hidden_size)
        self.cache = RealCache(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, embeddings):
        gate = self.router(embeddings[:, -1, :])
        
        # Calculate both outputs
        flow_out = self.flow(embeddings)
        cache_out = self.cache(embeddings)
        
        # THE HARD SWITCH: No mixing allowed!
        # If gate > 0.5, use cache. If gate < 0.5, use flow.
        # We expand gate to match the shape of the outputs for the math
        gate_expanded = gate.unsqueeze(-1) 
        
        # torch.where is PyTorch's version of a hard if/else statement
        final_output = torch.where(gate_expanded > 0.5, cache_out, flow_out)
        
        return final_output, gate

# ==========================================
# 3. TRAIN & VALIDATE PIPELINE
# ==========================================
def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    vocab_size = len(tokenizer)
    hidden_size = 768

    # --- STEP 1: TRAIN THE ROUTER ---
    print("--- STEP 1: TRAINING ROUTER ---")
    model_train = HardSwitchHybrid(hidden_size, vocab_size).cuda()
    
    # Freeze engines
    for p in model_train.flow.parameters(): p.requires_grad = False
    for p in model_train.cache.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.Adam(model_train.router.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    dataset = [
        ("The cat sat on the", 0.0),
        ("I went to the store to buy some", 0.0),
        ("To implement a binary search tree in", 1.0),
        ("The quantum superposition of the particle", 1.0),
        ("She opened the door and walked", 0.0),
    ]

    for epoch in range(10):
        for text, target_gate in dataset:
            inputs = tokenizer(text, return_tensors="pt").input_ids.cuda()
            with torch.no_grad(): embeddings = model_train.cache.lm.get_input_embeddings()(inputs)
            
            optimizer.zero_grad()
            _, gate_score = model_train(embeddings)
            target_tensor = torch.tensor([target_gate], dtype=torch.float32).cuda()
            loss = loss_fn(gate_score, target_tensor)
            loss.backward()
            optimizer.step()
    print("Training complete. Saving brain...\n")

    # --- STEP 2: BUILD FRESH MODEL & LOAD BRAIN ---
    print("--- STEP 2: LOADING TRAINED BRAIN ---")
    model_eval = HardSwitchHybrid(hidden_size, vocab_size).cuda()
    
    # Tie the weights so the mouth can speak English
    with torch.no_grad():
        model_eval.lm_head.weight = nn.Parameter(model_eval.cache.lm.get_input_embeddings().weight.clone())
    
    # LOAD THE BRAIN (Fixes the amnesia from Phase 4)
    model_eval.router.load_state_dict(model_train.router.state_dict())
    model_eval.eval() # Set to evaluation mode
    print("Brain loaded. Testing Hard Switch...\n")

    # --- STEP 3: TEST THE HARD SWITCH ---
    test_sentences = [
        ("The cat sat on the", "LOW GATE -> FLOW -> EXPECT GIBBERISH"),
        ("To implement a binary search tree in", "HIGH GATE -> CACHE -> EXPECT REAL WORD")
    ]
    
    for sentence, expectation in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            embeddings = model_eval.cache.lm.get_input_embeddings()(inputs)
            output_vector, gate_score = model_eval(embeddings)
            
            logits = model_eval.lm_head(output_vector)
            predicted_token_id = torch.argmax(logits, dim=-1)
            predicted_word = tokenizer.decode(predicted_token_id[0])
            
        print(f"Input: '{sentence}'")
        print(f"Expectation: {expectation}")
        print(f"Gate: {gate_score.item():.4f} | Predicted Word: '{predicted_word}'\n")

    print("VALIDATION COMPLETE.")

if __name__ == "__main__":
    main()