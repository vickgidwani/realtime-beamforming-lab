"""Compare synchronous and asynchronous batched beamforming on the T4.

Run from the repository root after building the patched Mach checkout:

    python scripts/benchmark_batched_beamforming.py
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import subprocess
import sys
from pathlib import Path
from beamforming_lab.streaming_beamformer import StreamingBeamformer
from beamforming_lab.streaming import load_cython_scheduler

import numpy as np

from trace_frame_replay import ROOT


MODES = ("overlapped", "async", "async_explicit_stream")


def beamform_power(runtime: StreamingBeamformer, batch_size: int, mode: str) -> np.ndarray:
    cp = runtime.cp
    beamform = runtime.beamform
    geometry = runtime.geometry
    workload = runtime.workload
    inputs = cp.asarray(
        np.ascontiguousarray(runtime.channel_data[:, :, :batch_size]),
        dtype=cp.complex64,
    )
    cp.cuda.Device().synchronize()
    kwargs = {
        "channel_data": inputs,
        "rx_coords_m": geometry.rx_coords_m,
        "scan_coords_m": geometry.scan_coords_m,
        "tx_wave_arrivals_s": geometry.tx_wave_arrivals_s,
        "f_number": float(workload.params["fnumber"]),
        "rx_start_s": float(workload.params["t0"]),
        "sampling_freq_hz": float(workload.params["fs"]),
        "sound_speed_m_s": float(workload.params["c"]),
        "modulation_freq_hz": float(workload.params["fc"]),
        "tukey_alpha": 0.0,
    }

    if mode == "async_explicit_stream":
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            image = beamform(**kwargs, synchronize=False, stream=stream)
            power = cp.sum(cp.abs(image) ** 2, axis=-1)
        stream.synchronize()
    elif mode == "async":
        with cp.cuda.Stream.null:
            image = beamform(**kwargs, synchronize=False)
            power = cp.sum(cp.abs(image) ** 2, axis=-1)
        cp.cuda.Stream.null.synchronize()
    else:
        image = beamform(**kwargs)
        power = cp.sum(cp.abs(image) ** 2, axis=-1)
        cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(power)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-hz", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=8_192)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exercise5-mach-batch-async-10khz.json"),
        help="JSON output path, relative to the repository root unless absolute",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Emit per-batch host staging, submit, release, and wait NVTX markers",
    )
    parser.add_argument(
        "--packer", choices=("strided", "one_pass"), default="strided",
        help="Select the host staging implementation",
    )
    parser.add_argument(
        "--concurrent-staging", action="store_true",
        help="Pack on a worker using a bounded pool of pinned host buffers",
    )
    parser.add_argument(
        "--reuse-beamform-output", action="store_true",
        help="Preallocate per-slot Mach outputs and arguments for explicit-stream replay",
    )
    parser.add_argument(
        "--trace-mach", action="store_true",
        help="Trace output preparation and native submission inside patched Mach",
    )
    parser.add_argument(
        "--prepare-mach", action="store_true",
        help="Validate and prepare one Mach launch per reusable device slot",
    )
    args = parser.parse_args()
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    modes = tuple(value.strip() for value in args.modes.split(","))
    if args.rate_hz <= 0 or args.frames <= 0 or args.queue_capacity <= 0:
        parser.error("rate, frames, and queue capacity must be positive")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        parser.error("batch sizes must be positive comma-separated integers")
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"modes must be selected from {', '.join(MODES)}")

    if args.reuse_beamform_output and modes != ("async_explicit_stream",):
        parser.error("--reuse-beamform-output requires --modes async_explicit_stream")
    if args.prepare_mach and not args.reuse_beamform_output:
        parser.error("--prepare-mach requires --reuse-beamform-output")
    if args.trace_mach and not args.trace_stages:
        parser.error("--trace-mach requires --trace-stages")

    os.chdir(ROOT)
    runtime = StreamingBeamformer()
    cp = runtime.cp
    frame_count = int(runtime.frame_count)
    full_replay_frames = int(runtime.replay_frames)
    if args.frames > full_replay_frames or args.frames % frame_count:
        parser.error(
            f"--frames must be a multiple of the {frame_count}-frame source "
            f"and no greater than {full_replay_frames}"
        )
    if any(size > frame_count for size in batch_sizes):
        parser.error(f"batch sizes may not exceed the {frame_count}-frame source")

    runtime.replay_frames = args.frames
    load_cython_scheduler()
    runtime.warmup(explicit_stream=True)
    cp.cuda.Device().synchronize()

    accuracy = {}
    for batch_size in batch_sizes:
        reference = beamform_power(runtime, batch_size, "overlapped")
        for mode in modes:
            actual = beamform_power(runtime, batch_size, mode)
            relative_rmse = float(
                np.linalg.norm(actual - reference)
                / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
            )
            accuracy[f"B{batch_size}/{mode}"] = relative_rmse

    results = []
    for batch_size in batch_sizes:
        for mode in modes:
            print(f"Running B={batch_size}, mode={mode} at {args.rate_hz:,} Hz", flush=True)
            result = runtime.run_batched_trial(
                batch_size,
                input_rate_hz=args.rate_hz,
                queue_capacity=args.queue_capacity,
                mode=mode,
                trace_stages=args.trace_stages,
                packer=args.packer,
                concurrent_staging=args.concurrent_staging,
                reuse_beamform_output=args.reuse_beamform_output,
                prepare_mach=args.prepare_mach,
                trace_mach=args.trace_mach,
            )
            row = {
                key: result[key]
                for key in (
                    "mode",
                    "batch_size_target",
                    "producer_kind",
                    "requested_rate_hz",
                    "offered_frames",
                    "accepted_frames",
                    "dropped_frames",
                    "batch_deadline_miss_frames",
                    "strict_frame_deadline_miss_frames",
                    "batch_deadline_failure_share_pct",
                    "strict_frame_failure_share_pct",
                    "trial_wall_s",
                    "completed_frames_per_s",
                    "latency_p95_ms",
                    "scheduled_age_p95_ms",
                    "batch_wall_p50_ms",
                    "host_stage_p50_ms_per_batch",
                    "h2d_event_p50_ms_per_batch",
                    "gpu_event_p50_ms_per_batch",
                    "batch_deadline_p50_ms",
                    "image_relative_rmse",
                )
            }
            results.append(row)
            print(json.dumps(row, indent=2), flush=True)

    output = {
        "experiment": "Exercise 5 batch plus asynchronous Mach comparison",
        "runtime": {
            "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
            "mach_version": version("mach-beamform"),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "configuration": {
            "requested_rate_hz": args.rate_hz,
            "offered_frames": args.frames,
            "queue_capacity_frames": args.queue_capacity,
            "batch_sizes": list(batch_sizes),
            "modes": list(modes),
            "producer_kind": "cython",
            "packer": args.packer,
            "concurrent_staging": args.concurrent_staging,
            "trace_stages": args.trace_stages,
            "reuse_beamform_output": args.reuse_beamform_output,
            "prepare_mach": args.prepare_mach,
            "trace_mach": args.trace_mach,
        },
        "batch_accuracy_relative_rmse_vs_synchronous": accuracy,
        "results": results,
    }
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print("\nComparison summary:")
    print(json.dumps(output, indent=2), flush=True)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
