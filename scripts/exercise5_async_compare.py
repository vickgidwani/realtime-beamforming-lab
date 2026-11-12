"""Compare synchronous and asynchronous Mach stream submission on Colab.

From the repository root, run the correctness and scheduled-throughput checks:

    python scripts/exercise5_async_compare.py

Capture a short CUDA/NVTX trace for the saturated 10 kHz trials:

    nsys profile --trace=cuda,nvtx,osrt --sample=none \
        --output=artifacts/exercise5-mach-async --force-overwrite=true \
        python scripts/exercise5_async_compare.py --trace-only

The runner executes only the notebook's configuration, runtime, data, and
implementation cells. Cython compilation and a CUDA warmup happen before the
trial ranges. The default run checks complete power maps at 200 Hz, then
compares the three submission paths at 10 kHz. The trace-only run profiles
only the latter comparison and does not overwrite result files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from exercise5_scheduled_trace import ROOT, load_notebook_runtime


BASE_MODES = ("overlapped", "synchronous_default", "async")
EXPLICIT_STREAM_MODE = "async_explicit_stream"


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    offered = int(result["offered_frames"])
    failures = int(result["dropped_frames"]) + int(result["deadline_misses"])
    return {
        "mode": result["mode"],
        "requested_rate_hz": result["requested_rate_hz"],
        "offered_frames": offered,
        "accepted_frames": result["accepted_frames"],
        "dropped_frames": result["dropped_frames"],
        "deadline_misses": result["deadline_misses"],
        "failure_fraction": failures / offered if offered else None,
        "trial_wall_s": result["trial_wall_s"],
        "accepted_frames_per_s": (
            result["accepted_frames"] / result["trial_wall_s"]
            if result["trial_wall_s"]
            else None
        ),
        "latency_p95_ms": result["latency_p95_ms"],
        "h2d_event_p50_ms": result["h2d_event_p50_ms"],
        "gpu_beamform_power_p50_ms": result["gpu_beamform_power_p50_ms"],
    }


def run_scenario(
    runtime: dict[str, Any], rate_hz: int, frames: int, modes: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    runtime["replay_frames"] = frames
    runs = {}
    for mode in modes:
        print(f"Running {mode}: {frames:,} frames at {rate_hz:,} Hz", flush=True)
        result = runtime["run_trial"](mode, rate_hz, producer_kind="cython")
        runs[mode] = result
        print(json.dumps(summarize(result), indent=2), flush=True)
    return runs


def relative_rmse(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(actual - reference))
    return float(np.linalg.norm(actual - reference) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-rate-hz", type=int, default=200)
    parser.add_argument("--accuracy-frames", type=int, default=512)
    parser.add_argument("--pipeline-rate-hz", type=int, default=10_000)
    parser.add_argument("--pipeline-frames", type=int, default=2_048)
    parser.add_argument("--include-explicit-stream", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    if min(args.accuracy_rate_hz, args.pipeline_rate_hz) <= 0:
        parser.error("rates must be positive")
    if min(args.accuracy_frames, args.pipeline_frames) <= 0:
        parser.error("frame counts must be positive")

    os.chdir(ROOT)
    runtime = load_notebook_runtime()
    cp = runtime["cp"]
    frame_count = int(runtime["frame_count"])
    full_replay_frames = int(runtime["replay_frames"])
    for label, frames in (
        ("--accuracy-frames", args.accuracy_frames),
        ("--pipeline-frames", args.pipeline_frames),
    ):
        if frames > full_replay_frames or frames % frame_count:
            parser.error(
                f"{label} must be a multiple of the {frame_count}-frame source "
                f"and no greater than {full_replay_frames}"
            )

    # Keep compilation and the first launch outside the measured trials.
    runtime["_load_cython_scheduler"]()
    runtime["_gpu_warmup"](explicit_stream=args.include_explicit_stream)
    cp.cuda.Device().synchronize()
    modes = BASE_MODES + ((EXPLICIT_STREAM_MODE,) if args.include_explicit_stream else ())

    output: dict[str, Any] = {
        "mach_async_prototype": True,
        "explicit_stream_prototype": args.include_explicit_stream,
        "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"],
        "mach_version": runtime["package_metadata"].version("mach-beamform"),
        "modes": list(modes),
        "source_frame_count": frame_count,
    }
    map_payload: dict[str, np.ndarray] = {}

    if not args.trace_only:
        accuracy_runs = run_scenario(runtime, args.accuracy_rate_hz, args.accuracy_frames, modes)
        reference = accuracy_runs["synchronous_default"]["power_map"]
        accuracy_summary = {}
        for mode, result in accuracy_runs.items():
            complete = int(result["accepted_frames"]) == args.accuracy_frames
            accuracy_summary[mode] = {
                "all_frames_accepted": complete,
                "relative_rmse_vs_synchronous_default": (
                    relative_rmse(result["power_map"], reference) if complete else None
                ),
            }
            map_payload[f"accuracy_{mode}"] = result["power_map"]
        output["accuracy_check"] = {
            "rate_hz": args.accuracy_rate_hz,
            "offered_frames": args.accuracy_frames,
            "reference_mode": "synchronous_default",
            "results": accuracy_summary,
        }

    pipeline_runs = run_scenario(runtime, args.pipeline_rate_hz, args.pipeline_frames, modes)
    output["scheduled_pipeline"] = {
        "rate_hz": args.pipeline_rate_hz,
        "offered_frames": args.pipeline_frames,
        "results": {mode: summarize(result) for mode, result in pipeline_runs.items()},
    }

    print("\nComparison summary:")
    print(json.dumps(output, indent=2, default=str), flush=True)
    if not args.trace_only:
        stem = "exercise5-mach-explicit-stream" if args.include_explicit_stream else "exercise5-mach-async-prototype"
        results_path = ROOT / "results" / f"{stem}.json"
        maps_path = ROOT / "results" / f"{stem}-maps.npz"
        results_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
        np.savez_compressed(maps_path, **map_payload)
        print(f"Saved: {results_path}")
        print(f"Saved: {maps_path}")


if __name__ == "__main__":
    main()
