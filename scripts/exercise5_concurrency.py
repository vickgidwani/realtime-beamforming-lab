"""Compare serial and two-stream B16 beamforming on the Exercise 5 workload.

Run on the Colab T4 from the repository root:

    python scripts/exercise5_concurrency.py benchmark
    nsys profile --trace=cuda,nvtx,osrt --sample=none \
        --output=artifacts/ex5-concurrency --force-overwrite=true \
        python scripts/exercise5_concurrency.py trace
    ncu --section LaunchStats --section Occupancy --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --kernel-name regex:beamformKernel --launch-count 1 \
        python scripts/exercise5_concurrency.py one-kernel

The input and geometry remain resident on the GPU. Each path processes the same
two B16 chunks, and each stream produces its own power map before the maps are
added. The test includes the contiguous chunk copy and power reduction, but not
RF-to-IQ conversion, geometry upload, or per-frame host-to-device replay.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
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


BATCH_SIZE = 16
FRAME_COUNT = 32


def load_inputs() -> tuple[Any, Any, Any, np.ndarray]:
    workload = load_pymust_doppler_workload(grid_size=251)
    geometry = prepare_geometry(workload, "cupy")
    channel_data = cp.asarray(workload.iq_elements_samples_frames)
    if channel_data.shape[2] != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} acquisitions; found {channel_data.shape[2]}")
    return workload, geometry, channel_data, workload.iq_elements_samples_frames


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
        )
        return cp.sum(cp.abs(beamformed) ** 2, axis=-1)


def run_serial(workload: Any, geometry: Any, channel_data: Any) -> tuple[float, float, np.ndarray]:
    stream = cp.cuda.Stream(non_blocking=True)
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    wall_start = time.perf_counter()
    RangePush("exercise5/concurrency/serial")
    start_event.record(stream)
    with stream:
        first = make_partial(workload, geometry, channel_data, 0, stream)
        second = make_partial(workload, geometry, channel_data, BATCH_SIZE, stream)
        power_map = first + second
        end_event.record(stream)
    end_event.synchronize()
    RangePop()
    gpu_ms = float(cp.cuda.get_elapsed_time(start_event, end_event))
    host_map = cp.asnumpy(power_map)
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    return gpu_ms, wall_ms, host_map


def run_concurrent(
    workload: Any, geometry: Any, channel_data: Any
) -> tuple[float, float, np.ndarray]:
    streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]
    join_stream = cp.cuda.Stream(non_blocking=True)
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    done_events = [cp.cuda.Event(), cp.cuda.Event()]
    wall_start = time.perf_counter()
    start_event.record(join_stream)
    for stream in streams:
        stream.wait_event(start_event)

    RangePush("exercise5/concurrency/two_streams")
    first = make_partial(workload, geometry, channel_data, 0, streams[0])
    second = make_partial(workload, geometry, channel_data, BATCH_SIZE, streams[1])
    for event, stream in zip(done_events, streams, strict=True):
        event.record(stream)
        join_stream.wait_event(event)
    with join_stream:
        power_map = first + second
        end_event.record(join_stream)
    end_event.synchronize()
    RangePop()
    gpu_ms = float(cp.cuda.get_elapsed_time(start_event, end_event))
    host_map = cp.asnumpy(power_map)
    wall_ms = (time.perf_counter() - wall_start) * 1_000
    return gpu_ms, wall_ms, host_map


def prepare_reference(workload: Any, geometry: Any) -> np.ndarray:
    reference = run_ensemble(workload, geometry, batch_size=BATCH_SIZE)["power_doppler"]
    return np.asarray(reference).ravel()


def run_one_kernel(workload: Any, geometry: Any, channel_data: Any) -> None:
    stream = cp.cuda.Stream(non_blocking=True)
    RangePush("exercise5/concurrency/ncu_one_beamform")
    make_partial(workload, geometry, channel_data, 0, stream)
    stream.synchronize()
    RangePop()


def run_trace(workload: Any, geometry: Any, channel_data: Any, reference: np.ndarray) -> None:
    serial = run_serial(workload, geometry, channel_data)
    concurrent = run_concurrent(workload, geometry, channel_data)
    print("trace serial GPU ms / wall ms:", round(serial[0], 4), round(serial[1], 4))
    print("trace two-stream GPU ms / wall ms:", round(concurrent[0], 4), round(concurrent[1], 4))
    for label, result in (("serial", serial), ("two_streams", concurrent)):
        rmse = np.linalg.norm(result[2].ravel() - reference) / np.linalg.norm(reference)
        print(f"{label} relative RMSE vs B16 reference: {rmse:.3e}")


def run_benchmark(
    workload: Any,
    geometry: Any,
    channel_data: Any,
    reference: np.ndarray,
    repeats: int,
) -> None:
    functions: dict[str, Callable[..., tuple[float, float, np.ndarray]]] = {
        "serial": run_serial,
        "two_streams": run_concurrent,
    }
    measurements: dict[str, list[tuple[float, float, np.ndarray]]] = {
        name: [] for name in functions
    }

    for _ in range(2):
        for function in functions.values():
            function(workload, geometry, channel_data)

    for iteration in range(repeats):
        order = ("serial", "two_streams") if iteration % 2 == 0 else ("two_streams", "serial")
        for name in order:
            measurements[name].append(
                functions[name](workload, geometry, channel_data)
            )

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(
        "GPU:",
        props["name"].decode(errors="replace"),
        "SMs:",
        props["multiProcessorCount"],
    )
    print("input shape:", tuple(channel_data.shape), "complex64; batch size:", BATCH_SIZE)
    for name, samples in measurements.items():
        gpu_ms = statistics.median(sample[0] for sample in samples)
        wall_ms = statistics.median(sample[1] for sample in samples)
        image = samples[-1][2].ravel()
        rmse = np.linalg.norm(image - reference) / np.linalg.norm(reference)
        print(
            f"{name}: median GPU={gpu_ms:.4f} ms, median wall={wall_ms:.4f} ms, "
            f"effective rate={FRAME_COUNT / (wall_ms / 1_000):.1f} acquisitions/s, "
            f"relative RMSE={rmse:.3e}"
        )
        print("  repeat wall ms:", [round(sample[1], 3) for sample in samples])

    serial_wall = statistics.median(sample[1] for sample in measurements["serial"])
    parallel_wall = statistics.median(sample[1] for sample in measurements["two_streams"])
    print("two-stream wall-time speedup:", round(serial_wall / parallel_wall, 3), "x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "trace", "one-kernel"))
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    workload, geometry, channel_data, _ = load_inputs()
    cp.cuda.Device().synchronize()
    print("dataset:", workload.data_provenance["dataset_id"])
    print("resident IQ:", tuple(channel_data.shape), "frame bytes:", channel_data[:, :, 0].nbytes)

    if args.mode == "one-kernel":
        run_one_kernel(workload, geometry, channel_data)
        return

    reference = prepare_reference(workload, geometry)
    if args.mode == "trace":
        run_trace(workload, geometry, channel_data, reference)
    else:
        run_benchmark(workload, geometry, channel_data, reference, args.repeats)


if __name__ == "__main__":
    main()
