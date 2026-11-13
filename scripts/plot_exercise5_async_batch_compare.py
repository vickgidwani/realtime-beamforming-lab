#!/usr/bin/env python3
"""Plot Exercise 5 per-frame async and batched async 10 kHz results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PER_FRAME_PATH = ROOT / "results" / "exercise5-mach-explicit-stream.json"
BATCH_PATH = ROOT / "results" / "exercise5-mach-batch-async-10khz.json"
OUTPUT_PATH = ROOT / "results" / "exercise5-async-batch-comparison.png"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


per_frame = load_json(PER_FRAME_PATH)["scheduled_pipeline"]["results"]
batch_doc = load_json(BATCH_PATH)
batch_rows = batch_doc["results"]

frame_modes = [
    ("overlapped", "Historical\noverlap", "#737b83"),
    ("synchronous_default", "Synchronous\ndefault", "#457b9d"),
    ("async", "Async\ndefault", "#2a9d8f"),
    ("async_explicit_stream", "Async explicit\nstream", "#9b5de5"),
]
batch_modes = [
    ("overlapped", "Sync Mach call", "#457b9d"),
    ("async", "Async default", "#2a9d8f"),
    ("async_explicit_stream", "Async explicit stream", "#9b5de5"),
]


def add_labels(ax, bars, values, fmt: str) -> None:
    top = max(values) if values else 1
    ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3, fontsize=8)
    ax.set_ylim(0, top * 1.18)


fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=False)
fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.21, hspace=0.5, wspace=0.23)
fig.suptitle("Exercise 5 · Async submission with and without batching", fontsize=16, fontweight="bold")

# Top row: per-frame paths. This is a 2,048-arrival replay.
for ax, metric, title, ylabel, fmt in [
    (axes[0, 0], "accepted_frames_per_s", "Per-frame: completed throughput", "Accepted frames/s", "{:.0f}"),
    (axes[0, 1], "latency_p95_ms", "Per-frame: p95 arrival-to-output", "Milliseconds", "{:.1f}"),
]:
    vals = [per_frame[mode][metric] for mode, _, _ in frame_modes]
    bars = ax.bar([label for _, label, _ in frame_modes], vals, color=[color for _, _, color in frame_modes])
    add_labels(ax, bars, vals, fmt)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)

# Bottom row: three paired submission modes at each batch target.
batch_sizes = sorted({row["batch_size_target"] for row in batch_rows})
x = np.arange(len(batch_sizes))
width = 0.23
batch_metrics = [
    (axes[1, 0], "completed_frames_per_s", "Batched: completed throughput", "Accepted frames/s", "{:.0f}"),
    (axes[1, 1], "latency_p95_ms", "Batched: p95 arrival-to-output", "Milliseconds", "{:.1f}"),
]
for ax, metric, title, ylabel, fmt in batch_metrics:
    maximum = 0.0
    for offset, (mode, label, color) in enumerate(batch_modes):
        vals = [
            next(row for row in batch_rows if row["batch_size_target"] == size and row["mode"] == mode)[metric]
            for size in batch_sizes
        ]
        maximum = max(maximum, max(vals))
        bars = ax.bar(x + (offset - 1) * width, vals, width, label=label, color=color)
        ax.bar_label(bars, labels=[fmt.format(value) for value in vals], padding=2, fontsize=7, rotation=90)
    ax.set_ylim(0, maximum * 1.29)
    ax.set_xticks(x, [f"B={size}" for size in batch_sizes])
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    if metric == "completed_frames_per_s":
        ax.legend(frameon=False, fontsize=8, loc="upper left")

axes[0, 0].text(0.99, 0.97, "2,048 offered frames", transform=axes[0, 0].transAxes,
                ha="right", va="top", fontsize=8, color="#555")
axes[1, 0].text(0.99, 0.97, "8,192 offered frames · queue capacity 8", transform=axes[1, 0].transAxes,
                ha="right", va="top", fontsize=8, color="#555")
fig.text(
    0.5,
    0.105,
    "All trials requested 10,000 frames/s. Every condition had 100% deadline-failure share; throughput counts accepted completions, not on-time frames.",
    ha="center",
    fontsize=9,
    color="#8b1e1e",
    fontweight="bold",
)
fig.text(
    0.5,
    0.065,
    "p95 is acquisition-arrival to output completion. Async improves batch throughput, while B=8 and B=16 p95 latency rises versus the same-run synchronous-call baseline.",
    ha="center",
    fontsize=8.5,
    color="#444",
)

fig.savefig(OUTPUT_PATH, dpi=200, facecolor="white")
print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
print("Same-run batched throughput gain over synchronous Mach call:")
for size in batch_sizes:
    baseline = next(row for row in batch_rows if row["batch_size_target"] == size and row["mode"] == "overlapped")
    for mode in ("async", "async_explicit_stream"):
        trial = next(row for row in batch_rows if row["batch_size_target"] == size and row["mode"] == mode)
        gain_pct = 100 * (trial["completed_frames_per_s"] / baseline["completed_frames_per_s"] - 1)
        print(f"  B={size} {mode}: {gain_pct:+.1f}%")
