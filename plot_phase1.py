"""
Phase 1 loss curve plot — all variants A, B, C, C-corrected, D, E.
Saves: plots/phase1_loss_curves.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs("plots", exist_ok=True)

VARIANTS = [
    ("A",      "logs/variant_a_loss.csv",           "#2196F3", "-",  "A — Pure GRU (2.204)"),
    ("B",      "logs/variant_b_loss.csv",           "#4CAF50", "-",  "B — Per-layer notepad (2.304)"),
    ("C-corr", "logs/variant_c_corrected_loss.csv", "#FF9800", "-",  "C-corr — Shared notepad corrected (2.523)"),
    ("D",      "logs/variant_d_loss.csv",           "#9C27B0", "-",  "D — Shared notepad + attention (2.322)"),
    ("E",      "logs/variant_e_loss.csv",           "#F44336", "--", "E — Transformer (3.266)"),
    ("C",      "logs/variant_c_naive_loss.csv",     "#795548", ":",  "C — Shared notepad naive (3.460)"),
]

WINDOW = 10   # rolling mean window for smoothing

fig, ax = plt.subplots(figsize=(11, 6))

for vid, path, color, ls, label in VARIANTS:
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue

    df = pd.read_csv(path)
    # deduplicate steps — keep last occurrence (handles Variant A crash/resume)
    df = df.drop_duplicates(subset="step", keep="last").sort_values("step")

    steps = df["step"].values
    loss  = df["loss"].values

    # raw — faint
    ax.plot(steps, loss, color=color, alpha=0.15, linewidth=0.8, linestyle=ls)

    # smoothed
    smooth = pd.Series(loss).rolling(window=WINDOW, min_periods=1, center=True).mean().values
    ax.plot(steps, smooth, color=color, linewidth=2.0, linestyle=ls, label=label)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss (nats)", fontsize=12)
ax.set_title("RiverNet Phase 1 — Loss Curves\n"
             "TinyStories · d=1024 · 6 layers · ~140–153M params · RTX 3050 4GB",
             fontsize=12)

ax.set_xlim(0, 5000)
ax.set_ylim(1.5, 7.0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout()
out = "plots/phase1_loss_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
