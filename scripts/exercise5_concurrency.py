"""Compare serial and two-stream B=8 beamforming on Exercise 5's full replay.

Run on the Colab T4 from the repository root:

    python scripts/exercise5_concurrency.py benchmark
    nsys profile --trace=cuda,nvtx,osrt --sample=none \
        --output=artifacts/exercise5-b8-two-compute-streams \
        --force-overwrite=true python scripts/exercise5_concurrency.py trace

Both paths process all 32 resident PyMUST IQ acquisitions as four B=8
beamformer calls. The parallel path distributes two calls to each of two
non-blocking compute streams, explicitly launches Mach with
``synchronize=False`` and the selected stream, then joins the streams before
combining power maps. This isolates compute concurrency from host-to-device
replay, RF-to-IQ conversion, and geometry upload. The trace mode warms both
paths before marking measured serial and two-stream ranges with NVTX.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cupy as cp
from cupy.cuda.nvtx import RangePop, RangePush
from mach.kernel import beamform

from beamforming_lab.beamformer import prepare_geometry, run_ensemble
from beamforming_lab.workload import load_pymust_doppler_workload


BATCH_SIZE = 8
FRAME_COUNT = 32
STREAM_COUNT = 2


def load_inputs() -> tuple[Any, Any, Any]:
    workload = load_pymust_doppler_workload(grid_size=251)
    geometry = prepare_geometry(workload, "cupy")
    channel_data = cp.asarray(workload.iq_elements_samples_frames)
    if channel_data.shape[2] != FRAME_COUNT:
        raise ValueError(
            f"Expected {FRAME_COUNT} acquisitions; found {channel_data.shape[2]}"
        )
    if FRAME_COUNT % BATCH_SIZE:
        raise ValueError("The replay frame count must be divisible by batch size")
    return workload, geometry, channel_data


def make_partial(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    first_frame: int,
    stream: Any,
) -> Any:
    with stream:
        chunk = cp.ascontiguousarray(
            channel_data[:, :, first_frame : first_frame + BATCH_SIZE]
        )
        beamformed = beamform(
            channel_data=chunk,
            rx_coords_m=geometry.rx_coords_m,
            scan_coords_m=geometry.scan_coords_m,
            tx_wave_arrivals_s=geometry.tx_wave_arrivals_s,
            f_number=float(workload.params["fnumber"]),
            rx_start_s=float(workload.params["t0"]),
            sampling_freq_hz=float(workload.params["fs"]),
            sound_speed_m_s=float(workload.params["c"]),
            modulation_freq_hz=float(workload.params["fc"]),
            tukey_alpha=0.0,
            synchronize=False,
            stream=stream,
        )
        return cp.sum(cp.abs(beamformed) ** 2, axis=-1)


def sum_partials(partials: list[Any], stream: Any) -> Any:
    if not partials:
        raise ValueError("At least one partial map is required")
    with stream:
        result = partials[0]
        for partial in partials[1:]:
            result = result + partial
    return result


def run_serial(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    *,
    annotate: bool = True,
) -> tuple[float, float, np.ndarray]:
    stream = cp.cuda.Stream(non_blocking=True)
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    wall_start = time.perf_counter()
    if annotate:
        RangePush("exercise5/concurrency/batch8/serial")
    try:
        start_event.record(stream)
        partials = [
            make_partial(workload, geometry, channel_data, first_frame, stream)
            for first_frame in range(0, FRAME_COUNT, BATCH_SIZE)
        ]
        power_map = sum_partials(partials, stream)
        end_event.record(stream)
        end_event.synchronize()
    finally:
        if annotate:
            RangePop()
    gpu_ms = float(cp.cuda.get_elapsed_time(start_event, end_event))
    host_map = cp.asnumpy(power_map)
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    return gpu_ms, wall_ms, host_map


def run_concurrent(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    *,
    annotate: bool = True,
) -> tuple[float, float, np.ndarray]:
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(STREAM_COUNT)]
    join_stream = cp.cuda.Stream(non_blocking=True)
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    done_events = [cp.cuda.Event() for _ in streams]
    stream_partials: list[list[Any]] = [[] for _ in streams]
    wall_start = time.perf_counter()
    if annotate:
        RangePush("exercise5/concurrency/batch8/two_compute_streams")
    try:
        start_event.record(join_stream)
        for stream in streams:
            stream.wait_event(start_event)

        for chunk_index, first_frame in enumerate(
            range(0, FRAME_COUNT, BATCH_SIZE)
        ):
            stream_index = chunk_index % STREAM_COUNT
            stream_partials[stream_index].append(
                make_partial(
                    workload,
                    geometry,
                    channel_data,
                    first_frame,
                    streams[stream_index],
                )
            )

        for event, stream in zip(done_events, streams, strict=True):
            event.record(stream)
            join_stream.wait_event(event)

        all_partials = [partial for lane in stream_partials for partial in lane]
        power_map = sum_partials(all_partials, join_stream)
        with join_stream:
            end_event.record(join_stream)
        end_event.synchronize()
    finally:
        if annotate:
            RangePop()
    gpu_ms = float(cp.cuda.get_elapsed_time(start_event, end_event))
    host_map = cp.asnumpy(power_map)
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    return gpu_ms, wall_ms, host_map


def prepare_reference(workload: Any, geometry: Any) -> np.ndarray:
    reference = run_ensemble(workload, geometry, batch_size=BATCH_SIZE)[
        "power_doppler"
    ]
    return np.asarray(reference)


def relative_rmse(image: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(image.ravel() - reference.ravel()) / np.linalg.norm(reference))


def warmup(
    workload: Any, geometry: Any, channel_data: Any
) -> None:
    run_serial(workload, geometry, channel_data, annotate=False)
    run_concurrent(workload, geometry, channel_data, annotate=False)
    cp.cuda.Device().synchronize()


def run_trace(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    reference: np.ndarray,
) -> None:
    warmup(workload, geometry, channel_data)
    serial = run_serial(workload, geometry, channel_data)
    concurrent = run_concurrent(workload, geometry, channel_data)
    for label, result in (("serial", serial), ("two_compute_streams", concurrent)):
        print(
            f"{label}: GPU={result[0]:.4f} ms, wall={result[1]:.4f} ms, "
            f"relative RMSE={relative_rmse(result[2], reference):.3e}"
        )
    print("two-stream wall-time speedup:", round(serial[1] / concurrent[1], 3), "x")


def run_benchmark(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    reference: np.ndarray,
    repeats: int,
    output_path: Path,
) -> None:
    functions: dict[str, Callable[..., tuple[float, float, np.ndarray]]] = {
        "serial": run_serial,
        "two_compute_streams": run_concurrent,
    }
    measurements: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in functions
    }

    warmup(workload, geometry, channel_data)
    for iteration in range(repeats):
        order = (
            ("serial", "two_compute_streams")
            if iteration % 2 == 0
            else ("two_compute_streams", "serial")
        )
        for name in order:
            gpu_ms, wall_ms, image = functions[name](
                workload, geometry, channel_data, annotate=False
            )
            measurements[name].append(
                (gpu_ms, wall_ms, relative_rmse(image, reference))
            )

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = properties["name"].decode(errors="replace")
    summaries: dict[str, Any] = {}
    for name, samples in measurements.items():
        gpu_median = statistics.median(sample[0] for sample in samples)
        wall_median = statistics.median(sample[1] for sample in samples)
        rmse_max = max(sample[2] for sample in samples)
        print(
            f"{name}: median GPU={gpu_median:.4f} ms, "
            f"median wall={wall_median:.4f} ms, "
            f"effective rate={FRAME_COUNT / (wall_median / 1_000):.1f} acquisitions/s, "
            f"max relative RMSE={rmse_max:.3e}"
        )
        print("  repeat GPU ms:", [round(sample[0], 4) for sample in samples])
        print("  repeat wall ms:", [round(sample[1], 4) for sample in samples])
        summaries[name] = {
            "gpu_ms_median": gpu_median,
            "wall_ms_median": wall_median,
            "effective_acquisitions_per_s": FRAME_COUNT / (wall_median / 1_000),
            "max_relative_rmse_vs_full_ensemble": rmse_max,
            "gpu_ms_samples": [sample[0] for sample in samples],
            "wall_ms_samples": [sample[1] for sample in samples],
            "relative_rmse_samples": [sample[2] for sample in samples],
        }

    speedup = summaries["serial"]["wall_ms_median"] / summaries[
        "two_compute_streams"
    ]["wall_ms_median"]
    print("two-stream wall-time speedup:", round(speedup, 3), "x")
    try:
        mach_version = version("mach-beamform")
    except PackageNotFoundError:
        mach_version = "unknown"
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "experiment": "Exercise 5 B=8 two-compute-stream capacity test",
        "runtime": {
            "device": device_name,
            "sm_count": int(properties["multiProcessorCount"]),
            "mach_version": mach_version,
            "git_revision": git_revision,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "configuration": {
            "dataset": workload.data_provenance["dataset_id"],
            "batch_size_per_stream_call": BATCH_SIZE,
            "frame_count": FRAME_COUNT,
            "beamform_calls_per_path": FRAME_COUNT // BATCH_SIZE,
            "concurrent_compute_streams": STREAM_COUNT,
            "warmup_rounds_per_path": 1,
            "measured_repeats_per_path": repeats,
            "input_and_geometry_resident_on_gpu": True,
            "includes_per_call_device_chunk_copy_and_power_reduction": True,
            "includes_host_to_device_replay": False,
        },
        "comparison": summaries,
        "two_stream_wall_time_speedup": speedup,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("saved results:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "trace"))
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results/exercise5-b8-two-compute-streams.json",
    )
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    workload, geometry, channel_data = load_inputs()
    cp.cuda.Device().synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print("GPU:", properties["name"].decode(errors="replace"))
    print("dataset:", workload.data_provenance["dataset_id"])
    print("resident IQ:", tuple(channel_data.shape), "complex64")
    print(
        f"each path: {FRAME_COUNT // BATCH_SIZE} calls of B={BATCH_SIZE}; "
        f"parallel path uses {STREAM_COUNT} asynchronous compute streams"
    )

    reference = prepare_reference(workload, geometry)
    if args.mode == "trace":
        run_trace(workload, geometry, channel_data, reference)
    else:
        run_benchmark(
            workload,
            geometry,
            channel_data,
            reference,
            args.repeats,
            args.output_json,
        )


if __name__ == "__main__":
    main()
