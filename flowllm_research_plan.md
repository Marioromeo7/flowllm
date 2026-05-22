# FlowLLM Research Plan
*A solo researcher's roadmap — 4GB VRAM, laptop only, no hype*

---

## Context

You built a 140M parameter GRU + shared gated notepad architecture (FlowLLM) targeting fixed VRAM regardless of context length. The architecture is a legitimate recombination of existing ideas (GRU, Neural Turing Machine, linear RNN theory) with one under-studied element: the shared single notepad across all layers outperforming per-layer memory.

**The goal is not to beat Claude. The goal is one clean, true, reproducible finding.**

---

## What's Actually Open in the Field

The field has Mamba, RWKV, and Griffin solving "no KV cache." What remains under-studied:

1. **Information capacity of fixed-size state** — no clean theoretical limit exists for what a fixed notepad can reliably compress vs. provably loses.
2. **Shared vs. hierarchical memory across layers** — your accidental finding that shared notepad beats per-layer is under-studied. Mamba has no external memory at all.
3. **Failure characterization of linear RNNs** — the "washing out" phenomenon you observed has no precise task-level characterization in the literature.

---

## Phase 1 — Nail the Ablation
**Timeline: 1–2 months**
**This is the only thing that matters right now.**

### What to build
Train 4 model variants to the same step count, everything else identical:

| Variant | Description |
|---|---|
| A | Pure GRU, no notepad |
| B | GRU + per-layer notepad |
| C | GRU + shared notepad *(yours)* |
| D | Transformer baseline, same parameter count |

### Rules
- Same dataset (TinyStories or equivalent)
- Same compute budget (steps, not wall time)
- Same evaluation metric (loss + a state-tracking task)
- **One controlled variable at a time — no exceptions**

### Why this first
Without this, you have observations, not findings. Every hour spent training longer before this is wasted.

---

## Phase 2 — Find a Task Where Yours Wins
**Timeline: 1 month**

Don't benchmark on general language quality. Find tasks that **structurally favor fixed memory**:

- **Long-range copy** — repeat something from 500 tokens ago
- **Multi-step arithmetic** — carry propagation across columns
- **State tracking** — door open/closed across a long narrative
- **Needle-in-haystack** — retrieve a specific fact from a long context

**Target**: your architecture beats Mamba on one of these at matched parameter count. That is a result worth writing up.

---

## Phase 3 — Characterize the Notepad's Information Capacity
**Timeline: 2 months**

This is the potentially novel contribution. The concrete experiment:

1. Vary notepad size: `[64d, 256d, 512d, 1024d]`
2. Hold everything else fixed
3. Measure what breaks as notepad shrinks
4. Attempt to derive a bound:

> *"A notepad of size N can reliably carry M bits of context across K tokens"*

This connects to existing **information bottleneck theory** and would be citable against real literature. Even a negative result (no clean bound exists) is publishable if the experiments are clean.

---

## Phase 4 — Write It Up Honestly
**Timeline: 1 month**

### Target title (working)
> *"Empirical characterization of fixed-size gated memory in recurrent language models: capacity limits and failure modes"*

### What to include
- Motivation and prior art (NTM, Mamba, RWKV — be precise)
- Architecture description with the bugs fixed (see below)
- Ablation results from Phase 1
- Task-specific results from Phase 2
- Capacity characterization from Phase 3
- Honest failure modes

### Target venue
ArXiv preprint. Not a top-tier conference submission — a clean, reproducible paper that someone else can build on.

---

## Known Bugs to Fix Before Any Further Training

| Bug | Location | Fix |
|---|---|---|
| `dimension` undefined | `FlowLLMBlock` MLP | Replace with `d_model` |
| `forward` nested inside `__init__` | `FlowLLM` | Fix indentation — method, not local function |
| Double `lm_head` call | `FlowLLM.forward` | Remove outer `lm_head()` wrapper |
| GRU hidden state discarded | `FlowLLMBlock.forward` | Thread `h_n` through generation like notepad |

**None of the Phase 1 results are valid until these are fixed.**

---

## What Would Kill the Project

- Training longer without ablations first — you'll get better at not knowing why
- Chasing generation quality before the science is done
- Comparing to Claude instead of Mamba/RWKV at matched parameter counts
- Claiming novelty before checking prior art per component

---

## Realistic Definition of Success

- An ArXiv preprint with clean ablations
- One finding such as: *"shared notepad outperforms per-layer by X% on state-tracking tasks at matched parameter count"*
- Code that someone else can clone and reproduce

That is a legitimate research contribution from a solo researcher on a laptop.

---

## Quick Reference: Prior Art to Cite

| Concept | Paper | Year |
|---|---|---|
| External memory with read/write gates | Neural Turing Machine (Graves et al.) | 2014 |
| Differentiable external memory | Differentiable Neural Computer (Graves et al.) | 2016 |
| Linear RNN at scale | RWKV (Peng et al.) | 2023 |
| State space models | Mamba (Gu & Dao) | 2023 |
| Gated linear attention | Griffin (De et al., Google DeepMind) | 2024 |
| Information bottleneck | Tishby et al. | 2000 |

---

## Runtime Estimates — Laptop Only, No Kaggle

*All runs on RTX 3050 Laptop, 4GB VRAM. Max sitting: ~1.5 hrs. Checkpoint every 500 steps.*
*Anchor: 500 steps took ~17 min on your machine (extrapolated from P100 + 2.7x slowdown).*

### The rule
**No single run goes more than 1.5 hours without a checkpoint pause.**
Every variant is split into ~1.5 hr chunks. Start a chunk, walk away, come back, resume.

---

### Phase 1 — Ablation (5,000 steps × 2 seeds, 140M model)

| Variant | Total time | Chunks of 1.5h | Days (1 chunk/day) |
|---|---|---|---|
| A — Pure GRU, no notepad | ~2.5 hrs | 2 | 2 |
| B — GRU + per-layer notepad | ~3.2 hrs | 3 | 3 |
| C — GRU + shared notepad | ~3.1 hrs | 3 | 3 |
| D — Transformer baseline | ~3.8 hrs | 3 | 3 |
| **TOTAL** | **~12.5 hrs** | **11 chunks** | **~11 days** |

> One chunk per day. Phase 1 completes in ~2 weeks at a relaxed pace.

---

### Phase 2 — Task benchmarks (5,000 steps × 2 seeds, 40M model)
*Smaller model = faster. All chunks fit comfortably under 1.5 hrs.*

| Task | Total time | Chunks | Days |
|---|---|---|---|
| Long-range copy (2 seeds) | ~1.4 hrs | 1 | 1 |
| Multi-step arithmetic (2 seeds) | ~1.4 hrs | 1 | 1 |
| State tracking (2 seeds) | ~1.4 hrs | 1 | 1 |
| **TOTAL** | **~4.2 hrs** | **3 chunks** | **3 days** |

> Phase 2 is 3 evenings. Smallest compute of the whole project.

---

### Phase 3 — Notepad capacity sweeps (5,000 steps × 2 seeds, extremes first)

| Run | Total time | Chunks | Days |
|---|---|---|---|
| d=64, seed 1 | ~2.5 hrs | 2 | 2 |
| d=64, seed 2 | ~2.5 hrs | 2 | 2 |
| d=1024, seed 1 | ~3.2 hrs | 3 | 3 |
| d=1024, seed 2 | ~3.2 hrs | 3 | 3 |
| d=256 + d=512 (only if extremes differ) | ~10 hrs | 7 | 7 |
| **TOTAL (extremes only)** | **~11.4 hrs** | **10 chunks** | **~10 days** |

> Start with d=64. If d=64 and d=1024 show no meaningful difference, stop — that is still a finding.

---

### Phase 4 — Writing

| Resource | Time |
|---|---|
| GPU compute | 0 hrs |
| Human time | 80–120 hrs across weeks |

---

### Grand total (laptop only, no Kaggle)

| Phase | GPU hrs | Days at 1 chunk/day |
|---|---|---|
| Phase 1 | ~12.5 hrs | ~11 days |
| Phase 2 | ~4.2 hrs | ~3 days |
| Phase 3 (extremes) | ~11.4 hrs | ~10 days |
| **TOTAL** | **~28 hrs** | **~24 days (~1 month)** |

**The entire compute phase fits in one calendar month at one 1.5-hour session per day.**

---

## Checkpoint Code — Add This to Every Training Script

Not optional. One crash without this = restart from zero.

```python
import os, csv, torch

CHECKPOINT_DIR = "checkpoints"
LOG_PATH       = "loss_log.csv"
SAVE_EVERY     = 500  # steps

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def init_log():
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss"])

def save_checkpoint(model, optimizer, step, loss):
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_step{step}.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss":      loss,
    }, path)
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([step, loss])
    print(f"[ckpt] step {step} | loss {loss:.4f} → {path}")

def load_latest_checkpoint(model, optimizer):
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split("step")[1].split(".")[0])
    )
    if not files:
        print("[ckpt] No checkpoint found. Starting fresh.")
        return 0
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[ckpt] Resumed from step {ckpt['step']} | loss {ckpt['loss']:.4f}")
    return ckpt["step"]

# ── Training loop template ──────────────────────────────────
# init_log()
# start_step = load_latest_checkpoint(model, optimizer)
#
# for step in range(start_step, TOTAL_STEPS):
#     loss = train_one_step(...)
#     if step % SAVE_EVERY == 0:
#         save_checkpoint(model, optimizer, step, loss.item())
```

### What this gives you
- Kill the script at any point — resume from the last 500-step mark
- `loss_log.csv` is your plot data for the paper, built automatically
- Keep only the last 3 checkpoint files — each ~550MB for 140M params

---

## Where FlowLLM Stands vs. What You Can Run Right Now

This section answers: *what is the best model that already exists for your hardware, and how far is FlowLLM from it?*

### What your hardware can actually run (4GB VRAM + 16GB RAM)

With 3–4GB VRAM, you can run 3–4B parameter models at Q4_K_M quantization with a 4K context window comfortably. The viable candidates for your exact setup are:

| Model | Params | VRAM (Q4_K_M) | Context | Tokens/sec (RTX 3050 est.) |
|---|---|---|---|---|
| **Qwen3-4B** | 4B | ~2.8 GB | 32K | ~12–18 tok/s |
| **Gemma 3 4B** | 4B | ~2.8 GB | 128K | ~12–18 tok/s |
| **Phi-4 Mini** | 3.8B | ~2.6 GB | 16K | ~15–20 tok/s |
| **Llama 3.2 3B** | 3B | ~2.1 GB | 128K | ~18–25 tok/s |
| **Gemma 3 1B** | 1B | ~0.8 GB | 32K | ~35–50 tok/s |

> The 8B class (Llama 3.1 8B, Qwen3 8B) needs ~6GB VRAM — it will partially offload to RAM on your machine, dropping to ~3–5 tok/s. Usable but painful.

---

### What these models can actually do

**Qwen3-4B** — current best-in-class at this size. Alibaba's Qwen 3 family is the model to beat at every parameter count from 4B to 32B. Solid reasoning, math, multilingual, thinking mode available.

**Gemma 3 4B** — Google's entry. One model for everything: code completions, commit message drafting, explaining stack traces, answering questions about unfamiliar libraries, writing docstrings — Gemma handles all of these more consistently than coding-specialized models when the task isn't pure code synthesis.

**Phi-4 Mini (3.8B)** — Microsoft's efficiency-first model. At 2.8 GB VRAM for completions, it leaves 5+ GB for your browser, IDE, running Docker containers, or a game in the background. Its accuracy is meaningfully better than what was possible at this parameter count in 2024 or 2025.

**Llama 3.2 3B** — fastest on your hardware, weakest reasoning of the group. Good for latency-sensitive tasks.

---

### How FlowLLM (140M) compares to these

Honest, no padding:

| Capability | Qwen3-4B / Gemma3-4B | FlowLLM 140M (current) |
|---|---|---|
| Parameters | 4,000M | 140M (28x smaller) |
| Training data | Trillions of tokens | ~500 steps on TinyStories |
| Coherent text generation | Yes | Story-shaped noise |
| Reasoning / math | Strong | None yet |
| MMLU benchmark | ~75%+ | Not measurable |
| VRAM at inference | ~2.8 GB | ~0.6 GB (fixed, no KV cache) |
| Context scalability | Grows with context (KV cache) | Fixed — does not grow |
| Tokens/sec on RTX 3050 | ~15 tok/s | ~60–80 tok/s (estimated, no KV overhead) |

**The one thing FlowLLM wins on right now: VRAM stays flat no matter the context length.** A 4B Transformer-based model at 32K context needs ~2.8GB weights + growing KV cache. FlowLLM at any context length needs only its weight footprint plus a fixed 1KB notepad.

That is the only honest advantage at this stage.

---

### How far is FlowLLM from Claude?

Claude Sonnet 4.6 (this model) runs on Anthropic's infrastructure with hundreds of billions of parameters, trained on effectively the entire internet for months on thousands of H100s.

| Dimension | Claude Sonnet 4.6 | FlowLLM 140M |
|---|---|---|
| Parameters | ~100B+ (estimated) | 140M |
| Training compute | Thousands of H100-months | ~500 steps on 1 GPU |
| Reasoning | Graduate-level | None |
| Coding | Production-grade | None |
| VRAM required | 100GB+ server VRAM | ~0.6 GB |
| Context window | 200K tokens | 256 tokens (rolling window) |

**The gap is not closeable at 140M parameters.** This is not a failure of the architecture — it is a parameter count and data problem. A 140M model trained to full convergence on a large dataset would be comparable to early GPT-2 (117M, 2019). Competent at syntax and simple patterns. Not capable of reasoning.

---

### What parameter count would make FlowLLM competitive with local 4B models?

Rough estimate, assuming the architecture is valid and training is done properly:

| FlowLLM scale | Comparable to | VRAM at inference | Feasible on your laptop? |
|---|---|---|---|
| 140M (current) | Early GPT-2 quality | ~0.6 GB | Train: yes. Run: yes. |
| 1B | Llama 3.2 1B class | ~0.9 GB | Train: no. Run: yes. |
| 3B | Phi-4 Mini class | ~1.5 GB | Train: no. Run: yes. |
| 7B | Llama 3.1 8B class | ~2.5 GB | Train: no. Run: yes (tight). |

Training a 1B+ FlowLLM from scratch would require cluster compute — weeks on multiple A100s. The research value of FlowLLM is not its scale, it is the **fixed-memory property at any scale**. A 3B FlowLLM running at 32K context with ~1.5GB flat VRAM would be genuinely useful and architecturally distinct from anything currently available.

That is a 2–3 year project from where you are now, assuming the ablations validate the approach.

---

### Practical recommendation

**Right now, today:** Download Qwen3-4B Q4_K_M via Ollama. Use it as your daily driver and as your qualitative benchmark for what "good" looks like at your hardware tier. It will show you concretely what FlowLLM needs to eventually match.

```bash
ollama pull qwen3:4b
ollama run qwen3:4b
```

**For FlowLLM:** The research path is valid. The architecture is not competitive yet — but competitive is not the goal. The goal is one clean finding about fixed-size memory. Do the ablations first.

---

*Last updated: May 2026*

---

## Parameter Scaling: Ceilings, Paths, and Honest Comparisons

*Added after Phase 4. This is the long-term view — only pursue after ablations validate the architecture.*

### The two ceilings

Training and inference have very different limits on your hardware. Don't conflate them.

**Training ceiling — ~200M parameters, hard stop.**
Training needs weights + gradients + Adam optimizer states + activations simultaneously. That's ~16 bytes per parameter. At 300M params you exceed your VRAM budget even with gradient checkpointing. Your current 140M model at ~2.3GB training VRAM is already close to the wall. The only knob left: reduce batch size to 1 and sequence length to 128, which might squeeze out ~220–240M. Beyond that, physics don't change.

**Inference ceiling — up to ~7B parameters (Q4 quantized).**
This is where FlowLLM's fixed-memory property actually matters. Because there is no KV cache, inference VRAM stays flat regardless of context length.

| Scale | fp16 VRAM | Q4 VRAM | Trainable on laptop? | Runnable on laptop? |
|---|---|---|---|---|
| 140M (current) | 0.28 GB | 0.07 GB | ✅ | ✅ |
| 300M | 0.60 GB | 0.15 GB | ❌ | ✅ |
| 1B | 2.00 GB | 0.50 GB | ❌ | ✅ |
| 3B | 6.00 GB | 1.50 GB | ❌ | ✅ (Q4 only) |
| 7B | 14.0 GB | 3.50 GB | ❌ | ✅ (Q4 only) |

A 7B FlowLLM at Q4 would fit in ~3.5GB flat. A 7B Transformer at 32K context needs ~6GB+ just for the KV cache on top of weights. That gap is the architectural claim in practice — not quality, memory scaling.

---

### Will a scaled FlowLLM compare to any existing pretrained model?

Honest answer by parameter count:

| FlowLLM scale | Comparable to | What that means |
|---|---|---|
| 140M fully trained | GPT-2 small (2019) | Coherent sentences, basic patterns, no reasoning |
| 300M fully trained | GPT-2 medium class | Slightly better prose, still no reasoning |
| 1B fully trained | Early LLaMA-1 1B class | Basic instruction following, weak reasoning |
| 3B fully trained | Phi-4 Mini / Llama 3.2 3B class | Actual reasoning, usable for simple tasks |
| 7B fully trained | Llama 3.1 8B class | Production-grade for constrained tasks |

The comparison that holds at **any** scale: a 3B FlowLLM at 32K context uses ~1.5GB VRAM flat. A 3B Transformer at 32K context needs ~1.5GB weights + ~3–4GB KV cache = ~5GB total and growing. That difference is real and is the only architectural claim worth making until ablations prove the notepad finding holds at scale.

FlowLLM at 140–200M params, fully trained, would be roughly GPT-2 small quality: coherent sentences, no reasoning. That is the honest ceiling for what your hardware can train.

---

### The realistic scaling paths

**Path A — Train 140M to full convergence (laptop, free)**
Do this after the ablations. 3–5 epochs on TinyStories or FineWeb-Edu-10B sample on Kaggle across many sessions. The resulting model is GPT-2 class. Publishable as a proof of concept for the architecture. This is the natural continuation of the current plan.

**Path B — Train 1B FlowLLM on a rented GPU (~$30–50)**
One A100 session on Lambda Labs or vast.ai. A 1B FlowLLM trained properly would be early LLaMA-1 1B class. That is a model you can concretely compare against published benchmarks. Only pursue this if Phase 1–3 ablations show the shared notepad finding holds.

**Path C — Train 3B FlowLLM (~$200–400 rented compute)**
This is where FlowLLM becomes genuinely useful and architecturally distinct. A 3B model running at 32K context in 1.5GB flat VRAM has no equivalent in the current open-source landscape. This is a 1–2 year goal from where you are now.

---

### What would make a scaled FlowLLM worth training

Before spending money on rented compute, the ablations must show:

1. Shared notepad outperforms per-layer notepad on at least one structured task
2. The notepad genuinely compresses information across context (Phase 3 finding)
3. Loss continues to drop past the current 4.6 plateau with more training

If all three hold, a 1B FlowLLM trained on a real dataset becomes a legitimate research artifact. If they don't hold, scaling would just produce a larger version of the same limitations — and a different architecture would be warranted.

**The ablations are the gate. Nothing in this section matters until they're done.**

---

*Last updated: May 2026*
