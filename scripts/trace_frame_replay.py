"""Run one Exercise 5 scheduled trial for an Nsight Systems trace.

From the repository root on the Colab T4, profile the short overlapped replay:

    nsys profile --trace=cuda,nvtx,osrt --sample=none \
        --output=artifacts/exercise5-scheduled-overlapped-10khz \
        --force-overwrite=true \
        python scripts/trace_frame_replay.py --mode overlapped

Then restrict CLI summaries to the marked trial range:

    nsys stats --filter-nvtx exercise5/cython/overlapped/10000Hz \
        --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
        artifacts/exercise5-scheduled-overlapped-10khz.nsys-rep

The runner imports the same replay module used by the notebook. Cython compilation and a CUDA warmup happen before the
NVTX capture range. By default the trace contains 2,048 offered arrivals at
10 kHz (64 repeats of the 32-frame source), enough to show sustained queueing
and GPU gaps without profiling the full rate matrix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from beamforming_lab.streaming_beamformer import StreamingBeamformer, QUEUE_CAPACITY
from beamforming_lab.streaming import load_cython_scheduler


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serialized", "overlapped"), default="overlapped")
    parser.add_argument("--rate-hz", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=2_048)
    args = parser.parse_args()
    if args.rate_hz <= 0 or args.frames <= 0:
        parser.error("--rate-hz and --frames must be positive")

    os.chdir(ROOT)
    runtime = StreamingBeamformer()
    cp = runtime.cp
    frame_count = runtime.frame_count
    full_replay_frames = runtime.replay_frames
    if args.frames > full_replay_frames or args.frames % frame_count:
        parser.error(
            f"--frames must be a multiple of the {frame_count}-frame source "
            f"and no greater than the full replay ({full_replay_frames})"
        )

    # Keep setup and warmup outside the trial range used for filtered analysis.
    load_cython_scheduler()
    runtime.warmup()
    cp.cuda.Device().synchronize()
    runtime.replay_frames = args.frames

    print(
        f"Starting scheduled trace: mode={args.mode}, rate={args.rate_hz} Hz, "
        f"offered={args.frames}, queue_capacity={QUEUE_CAPACITY}"
    )
    result = runtime.run_trial(args.mode, args.rate_hz, producer_kind="cython")
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "producer": result["producer_kind"],
                "requested_rate_hz": result["requested_rate_hz"],
                "offered_frames": result["offered_frames"],
                "accepted_frames": result["accepted_frames"],
                "dropped_frames": result["dropped_frames"],
                "deadline_misses": result["deadline_misses"],
                "trial_wall_s": result["trial_wall_s"],
                "gpu_beamform_power_p50_ms": result["gpu_beamform_power_p50_ms"],
                "h2d_event_p50_ms": result["h2d_event_p50_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
