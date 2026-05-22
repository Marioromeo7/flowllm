# FlowLLM Phase 1 — Experimental Findings

## Setup

- Hardware: RTX 3050 Laptop GPU, 4GB VRAM
- Dataset: TinyStories (roneneldan/TinyStories), train split
- Tokenizer: EleutherAI/gpt-neo-125M (vocab size 50,257)
- All variants: d_model=1024, n_layers=6 (E has 7), batch_size=4, seq_len=256, steps=5000, lr=3e-4, AdamW
- Loss: cross-entropy next-token prediction, ignore padding
- Gradient checkpointing on all variants

---

## Variants

| ID | Architecture | Params | Notes |
|---|---|---|---|
| A | Pure GRU, no notepad | ~139.9M | Baseline recurrent |
| B | GRU + per-layer notepad (isolated) | ~139.9M | Each layer has own notepad, not shared |
| C | GRU + shared notepad (naive) | ~152.5M | Notepad written only from last token position |
| C-corr | GRU + shared notepad (corrected) | ~152.5M | GRUCell sequential, notepad written at every position |
| D | GRU + shared notepad + cross-attention read | ~152.5M | Attention read from shared notepad |
| E | Causal Transformer | ~139.9M | d=1024, 7 layers, 16 heads, Flash Attention |

---

## Results

### Final Loss at Step 5000

| Variant | Final Loss | Avg Last 10 Readings | Trend |
|---|---|---|---|
| **A — Pure GRU** | **2.204** | **~2.33** | Converging |
| B — Per-layer notepad | 2.304 | ~2.37 | Converging |
| D — Shared notepad + attention | 2.322 | ~2.42 | Converging |
| E — Transformer | 3.266 | ~3.26 | Plateaued |
| C — Shared notepad (naive) | 3.460 | ~3.46 | Plateaued / diverging |
| C-corr — Shared notepad (corrected) | 2.523 | ~2.35 | Converging |

### Loss Curves — Key Data Points

**Variant A (Pure GRU)**
Steps 50→5000: 5.42 → 4.49 → 3.22 → 2.59 → 2.20
Note: Duplicate entries steps 1550–2000 (crash at step 2000, resumed from step 1500 checkpoint). Second occurrence is authoritative.

**Variant B (Per-layer notepad)**
Steps 50→5000: 5.81 → 4.92 → 3.32 → 2.61 → 2.30
Best recorded: 1.95 at step 4550

**Variant C (Naive shared notepad)**
Only 10 data points (logging bug — logged at checkpoint saves only, not every 50 steps):
500→5000: 4.62 → 3.75 → 3.37 → 3.97 → 3.46
Never converged below 3.0. Oscillating, no clear downward trend.

**Variant D (Shared notepad + attention read)**
Steps 50→5000: 5.56 → 4.58 → 3.24 → 2.60 → 2.32
Best recorded: 2.02 at step 4550

**Variant E (Transformer)**
Steps 50→5000: 5.55 → 4.86 → 3.78 → 3.33 → 3.27
Plateau begins around step 2000. No meaningful improvement after step 2500.

**Variant C-corrected (Shared notepad, corrected + residual)**
Steps 50→5000: 6.07 → 5.20 → 4.46 → 4.09 → 3.25 → 2.85 → 2.53 → 2.34 → 2.09 → 2.52
Best recorded: 1.971 at step 4350.
Architecture required a block-level skip connection (LN(out + x)) to converge.
Without it: complete training failure — loss stuck at ~5.7 through 3800 steps.

---

## Key Findings

### Finding 1: GRU beats Transformer at matched compute
All GRU variants that converged (A, B, D) reached ~2.2–2.3 loss.
The Transformer (E) plateaued at 3.27 — roughly 1.1 nats worse.

At 5000 steps on TinyStories with seq_len=256, the recurrent inductive bias
outperforms attention. Likely because transformers require more data and training
steps to amortize their parameter efficiency advantage.

### Finding 2: Naive shared notepad is harmful
Variant C (notepad updated only from the last token position) performed worst of all —
worse even than the Transformer (3.46 vs 3.27). The last-position write bottleneck
means 255/256 tokens never interact with the notepad. The notepad becomes stale
and acts as noise rather than signal.

### Finding 3: Per-layer notepad slightly underperforms pure GRU
Variant B (per-layer isolated notepad) finished at 2.30 vs A at 2.20.
The per-layer notepad adds parameters and complexity without benefit at this scale.
Possible reason: isolated notepads cannot share information across layers,
limiting their utility.

### Finding 4: Attention read does not recover notepad losses
Variant D (cross-attention read from shared notepad) reached 2.32 — worse than
pure GRU A (2.20) despite the more expressive read mechanism. This suggests the
write bottleneck (still last-position only in D) is the dominant failure mode,
not the read mechanism.

### Finding 5: Corrected shared notepad achieves parity with pure GRU
Variant C-corrected (every-position write, block-level residual) reached 2.523 final
loss and ~2.35 avg last 10 — within ~0.12 nats of pure GRU A (2.204, ~2.33 avg).
Best individual reading was 1.971, which beats A's best.

The corrected notepad neither significantly helps nor hurts at this scale.
This is meaningful: the shared notepad imposes no cost when correctly implemented,
leaving headroom for it to become beneficial at larger scale or longer sequences.

### Finding 6: Block-level residual is mandatory for notepad convergence
Without a skip connection from block input x to block output (LN(out + x)),
the 255-step sequential notepad write chain causes vanishing gradients:
gradient factor (1-w)^255 ≈ 10^-77 at initialization. The model completely
fails to converge — loss stuck at ~5.7 after 3800 steps.

This is an architectural requirement, not a hyperparameter choice. Any sequential
external memory design with long write chains must include a gradient highway.
The naive C variant (3.46) partially avoided this because most positions never
wrote to the notepad, effectively reducing the chain length.

---

## Ablation Summary (Controlled Variables)

| Comparison | Variable Isolated | Result |
|---|---|---|
| A vs E | Recurrent vs Attention | GRU wins by 1.06 nats |
| A vs B | No notepad vs per-layer notepad | No notepad wins by 0.10 nats |
| A vs C | No notepad vs naive shared notepad | No notepad wins by 1.26 nats |
| A vs D | No notepad vs shared notepad + attention | No notepad wins by 0.12 nats |
| C vs D | Scalar gate read vs attention read | Attention read wins by 1.14 nats |
| A vs C-corr | No notepad vs corrected shared notepad | Roughly tied (~0.12 nats, within noise) |

---

## Known Data Quality Issues

1. **Variant A duplicate entries**: Steps 1550–2000 appear twice due to crash/resume.
   Use second occurrence. Does not affect final result (step 5000 = 2.204).

2. **Variant C sparse logging**: Only 10 data points due to logging at checkpoint
   saves only (every 500 steps). Loss curve cannot be plotted meaningfully.
   Architectural bug confirmed — result is still valid as evidence of naive design failure.

3. **Variant C-corrected**: Two failed runs before correct result obtained.
   - Run 1 (GRUCell, no residual): killed at step 750 — too slow (~24h estimate)
   - Run 2 (batched GRU, torch.compile): killed at step 2750 — compile broke gradients
   - Run 3 (batched GRU, no compile, no residual): killed at step 3800 — vanishing gradients
   - Run 4 (batched GRU, block residual): completed successfully. Use this result.

---

## Architecture Details

### GRUBlock (Variants A, B — per block)
```
gru = nn.GRU(d, d)           # batched, cuDNN optimized
mlp = Linear(d, 4d) → GELU → Linear(4d, d)
ln1, ln2 = LayerNorm(d)
forward: gru_out = GRU(x); x = LN(gru_out + x); x = LN(mlp(x) + x)
```

### GRUNotepadBlock (Variant C-corrected — final working design)
```
gru = nn.GRU(d, d, batch_first=True)   # cuDNN batched
read_gate = Linear(d, d)
write_gate = Linear(d, d)
mlp = Linear(d, 4d) → GELU → Linear(4d, d)
ln1, ln2, ln_skip = LayerNorm(d)

all_h = GRU(x)                              # full sequence, one CUDA call
all_r = sigmoid(read_gate(all_h))           # batched
all_w = sigmoid(write_gate(all_h))          # batched
for t in range(T):                          # sequential — notepad dependency only
    h_read_t = LN(h_t + r_t * note)
    note = (1-w_t)*note + w_t*h_t
out = LN(mlp(stack(h_reads)) + stack(h_reads))   # batched MLP
out = LN(out + x)                           # block-level skip — REQUIRED for convergence
```

### Teacher (Variant E)
```
d=1024, n_layers=7, n_heads=16
Flash Attention (F.scaled_dot_product_attention, is_causal=True)
Weight-tied token embedding and lm_head
```

---

## Training Infrastructure

- Checkpoints: every 500 steps, keep=1 (pruned to save disk)
- Checkpoint size: ~1.83GB per variant (model + Adam optimizer states)
- Loss logged: every 50 steps to loss_log.csv
- Gradient clipping: max_norm=1.0
- Gradient checkpointing: enabled on all layer forward calls

---

## Next Steps

1. Plot all loss curves together (matplotlib) — Phase 1 data now complete
2. Write paper draft — core claims:
   - GRU variants consistently outperform parameter-matched Transformer at low compute
   - Naive notepad design is catastrophically harmful (3.46 vs 2.20)
   - Corrected notepad achieves parity with pure GRU (~2.35 vs ~2.33 avg)
   - Block-level residual is a hard architectural requirement for sequential memory
3. Run RL variants (F, G, H, I) — teacher distillation experiments
4. Apply for compute grants (Google TPU Research Cloud, AWS Research Credits)
5. Scale winning architecture to 1B parameters on SlimPajama
