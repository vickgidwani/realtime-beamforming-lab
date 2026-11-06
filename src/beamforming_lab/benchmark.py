from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .beamformer import prepare_geometry, run_ensemble
from .provenance import runtime_provenance
from .workload import Workload


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(array.max()),
    }


def benchmark(
    workload: Workload,
    *,
    mode: str,
    runs: int,
    warmup: int,
    batch_size: int,
    deadline_ms: float | None,
    measure_cold_start: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup nonnegative")

    frame_count = workload.iq_elements_samples_frames.shape[2]
    prf_hz = float(workload.params["PRF"])
    acquisition_window_ms = frame_count / prf_hz * 1000.0

    cold_start_measurement = None
    cold_start_started_ns = 0
    geometry_prepare_start_ns = 0
    if measure_cold_start:
        cold_start_started_ns = time.perf_counter_ns()
        geometry_prepare_start_ns = cold_start_started_ns
    geometry = prepare_geometry(workload, mode)
    if measure_cold_start:
        geometry_prepare_wall_ms = (
            (time.perf_counter_ns() - geometry_prepare_start_ns) / 1_000_000
        )
        first_result = run_ensemble(workload, geometry, batch_size=batch_size)
        first_ensemble_wall_ms = float(first_result["run_wall_ms"])
        setup_to_first_output_ms = (
            (time.perf_counter_ns() - cold_start_started_ns) / 1_000_000
        )
        cold_start_measurement = {
            "geometry_prepare_wall_ms": float(geometry_prepare_wall_ms),
            "first_ensemble_pipeline_wall_ms": first_ensemble_wall_ms,
            "device_setup_plus_first_ensemble_wall_ms": float(
                setup_to_first_output_ms
            ),
            "first_sample_to_output_ms": float(
                acquisition_window_ms
                + workload.rf_to_iq_ms
                + setup_to_first_output_ms
            ),
            "scope": (
                "Starts immediately before prepare_geometry and ends after the first "
                "Power Doppler map is ready. For NumPy, first CUDA use occurs inside "
                "the first mach call; for CuPy, it occurs during geometry preparation. "
                "Dataset loading and RF-to-IQ conversion happen before this timer. "
                "The cold call is separate from warmup and measured latency samples."
            ),
        }

    for _ in range(warmup):
        run_ensemble(workload, geometry, batch_size=batch_size)

    run_wall: list[float] = []
    input_upload: list[float] = []
    beamform_host: list[float] = []
    beamform_device: list[float] = []
    output_features: list[float] = []
    latest_power = None
    for _ in range(runs):
        result = run_ensemble(workload, geometry, batch_size=batch_size)
        run_wall.append(result["run_wall_ms"])
        input_upload.append(result["input_upload_ms"])
        beamform_host.append(result["beamform_host_call_ms"])
        if result["beamform_device_ms"] is not None:
            beamform_device.append(result["beamform_device_ms"])
        output_features.append(result["output_feature"])
        latest_power = result["power_doppler"]

    first_sample_latency = [
        acquisition_window_ms + workload.rf_to_iq_ms + duration
        for duration in run_wall
    ]
    miss_count = 0
    if deadline_ms is not None:
        miss_count = sum(latency > deadline_ms for latency in first_sample_latency)

    report = {
        "dataset": "PyMUST rotating-disk Doppler raw-RF phantom",
        "data_provenance": workload.data_provenance,
        "runtime_provenance": runtime_provenance(),
        "array_mode": mode,
        "iq_shape_elements_samples_frames": list(
            workload.iq_elements_samples_frames.shape
        ),
        "scan_grid_points": int(workload.scan_coords_m.shape[0]),
        "frames_per_ensemble": frame_count,
        "prf_hz": prf_hz,
        "acquisition_window_ms": acquisition_window_ms,
        "rf_to_iq_ms_once": workload.rf_to_iq_ms,
        "static_geometry_upload_ms_once": geometry.geometry_upload_ms,
        "batch_size_frames": batch_size,
        "warmup_runs": warmup,
        "measurement_runs": runs,
        "cold_start_measurement": cold_start_measurement,
        "deadline_ms": deadline_ms,
        "deadline_misses_first_sample_to_output": miss_count,
        "deadline_miss_rate": miss_count / runs if deadline_ms is not None else None,
        "first_sample_to_output_ms": _distribution(first_sample_latency),
        "ensemble_pipeline_wall_ms": _distribution(run_wall),
        "per_ensemble_input_upload_ms": _distribution(input_upload),
        "beamform_host_call_sum_ms": _distribution(beamform_host),
        "beamform_device_event_sum_ms": _distribution(beamform_device),
        "equivalent_beamformed_frames_per_second": frame_count
        / (float(np.median(run_wall)) / 1000.0),
        "mean_output_feature": float(np.mean(output_features)),
        "notes": [
            "Power Doppler output is computed from the beamformed ensemble.",
            "The first-sample latency estimate serializes acquisition, RF-to-IQ, and measured processing; overlap can reduce it.",
            "The 32-frame raw-RF ensemble is repeated to estimate latency percentiles.",
            "In cupy mode, IQ is uploaded once per ensemble; geometry stays GPU-resident across runs.",
        ],
    }
    return report, latest_power


def save_report(report: dict[str, Any], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
