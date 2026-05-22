# FlowLLM Project — Claude Context

## Project Goal

Research project: characterize fixed-size gated memory in recurrent language models.
Target output: ArXiv preprint with clean ablations.
**The goal is one clean, reproducible finding — not to beat Claude.**

## Hardware

- RTX 3050 Laptop GPU, **4GB VRAM**
- Max single training run: ~1.5 hours (checkpoint every 500 steps)
- Conda environment: `hybrid_router`

## Architecture Summary

FlowLLM = GRU layers + shared single notepad (fixed-size external memory).
Key finding under investigation: shared notepad across all layers outperforms per-layer memory.
VRAM stays flat regardless of context length — no KV cache.

## File Map

| File | What it is |
|---|---|
| `flow_llm_300m.py` | Main 140M FlowLLM — `FlowLLM` + `FlowLLMBlock` on TinyStories |
| `flow_with_notepad.py` | `FlowNotepadModel` — single GRUCell + gated notepad (math task) |
| `pure_flow_baseline.py` | `PureFlowModel` — 2-layer GRU, no notepad (ablation variant A) |
| `baseline_transformer.py` | `PureTransformer` — causal transformer (ablation variant D) |
| `hybrid_challenger.py` | `HybridModel` — GRU + sliding-window attention with soft router |
| `phase1_router.py` | Skeleton `Hybrid3050Killer` with fake engines (proof of concept) |
| `phase5_hard_switch.py` | `HardSwitchHybrid` — hard switch GRU vs distilGPT2 |
| `flow_stress_test.py` | Stress test: 1-digit → 3-digit math generalization |
| `flowllm_research_plan.md` | Full research roadmap (read this first) |

## Ablation Variants (Phase 1)

| Variant | Model | File |
|---|---|---|
| A | Pure GRU, no notepad | `pure_flow_baseline.py` |
| B | GRU + per-layer notepad | (to build) |
| C | GRU + shared notepad | `flow_llm_300m.py` |
| D | Transformer baseline | `baseline_transformer.py` |

## Known Bugs in `flow_llm_300m.py` — Fix Before Any Training

1. **GRU hidden state discarded** — `FlowLLMBlock.forward` passes `torch.zeros` each call instead of threading `h_n` through generation
2. **ManualGRU weight indexing** — line 59 references `self.xh_to_gates.weight` with wrong slicing (ManualGRU is currently unused but broken)

Check `flowllm_research_plan.md` § "Known Bugs" for the full list before training.

## Research Phases

| Phase | Goal | Timeline |
|---|---|---|
| 1 | Ablation: A vs B vs C vs D | ~11 days at 1 chunk/day |
| 2 | Task benchmarks (long-range copy, arithmetic, state tracking) | ~3 days |
| 3 | Notepad capacity sweep (d=64, 256, 512, 1024) | ~10 days |
| 4 | Write up | 80–120 human hours |

**Phase 1 ablations are the gate. Nothing else matters until they're done.**

## Checkpoint Convention

- Save every 500 steps to `checkpoints/ckpt_step{N}.pt`
- Log loss to `loss_log.csv`
- Keep only last 3 checkpoints (~550MB each for 140M params)
- Template in `flowllm_research_plan.md` § "Checkpoint Code"

## Training Rules

- No single run > 1.5 hours without a checkpoint pause
- Same dataset, same compute budget, same eval metric across all ablation variants
- One controlled variable at a time
- Dataset: TinyStories (or equivalent) for Phase 1

## What to Avoid

- Training longer without ablations first
- Comparing to Claude (compare to Mamba/RWKV at matched parameter count)
- Claiming novelty before checking prior art per component
- Adding features before the science is done
