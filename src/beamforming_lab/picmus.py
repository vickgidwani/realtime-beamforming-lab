from __future__ import annotations

import contextlib
import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PICMUS_URL = (
    "https://zenodo.org/records/20261898/files/"
    "PICMUS_experiment_resolution_distortion.uff"
)
PICMUS_FILE = "PICMUS_experiment_resolution_distortion.uff"
PICMUS_SIZE_BYTES = 145_518_524
PICMUS_SHA256 = "c93af0781daeebf771e53a42a629d4f311407410166ef2f1d227e9d2a1b8c641"
ARRAY_ARGUMENTS = ("channel_data", "rx_coords_m", "scan_coords_m", "tx_wave_arrivals_s")


@dataclass(frozen=True)
class PicmusWorkload:
    beamform_args: dict[str, Any]
    grid_shape: tuple[int, int]
    setup_ms: float
    provenance: dict[str, Any]


def load_picmus_workload() -> PicmusWorkload:
    """Download/cache PICMUS and prepare the multi-angle mach input arrays."""
    setup_start_ns = time.perf_counter_ns()
    try:
        from mach.io.uff import create_beamforming_setup
        from mach.io.utils import cached_download
        from pyuff_ustb import Uff
    except ImportError as exc:
        raise RuntimeError(
            "PICMUS loading requires the mach example dependencies. Install "
            "with `python -m pip install -e '.[gpu]'`."
        ) from exc

    data_path: Path = cached_download(
        PICMUS_URL,
        expected_size=PICMUS_SIZE_BYTES,
        expected_hash=PICMUS_SHA256,
        digest=hashlib.sha256,
        filename=PICMUS_FILE,
    )
    uff_file = Uff(str(data_path))
    channel_data = uff_file.read("/channel_data")
    scan = uff_file.read("/scan")
    beamform_args = create_beamforming_setup(
        channel_data=channel_data,
        scan=scan,
        f_number=1.7,
    )

    # Make layout conversion explicit and one-time so the timed runs measure
    # the host/device path and beamforming rather than a hidden stride repair.
    beamform_args = {
        key: np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        for key, value in beamform_args.items()
    }

    channel_array = np.asarray(beamform_args["channel_data"])
    tx_angles_deg = [
        float(np.rad2deg(wave.source.azimuth)) for wave in channel_data.sequence
    ]
    authorship = ""
    print_authorship = getattr(channel_data, "print_authorship", None)
    if callable(print_authorship):
        authorship_stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(authorship_stream):
                print_authorship()
            authorship = authorship_stream.getvalue().strip()
        except Exception:
            authorship = ""

    setup_ms = (time.perf_counter_ns() - setup_start_ns) / 1_000_000
    return PicmusWorkload(
        beamform_args=beamform_args,
        grid_shape=(int(scan.x_axis.size), int(scan.z_axis.size)),
        setup_ms=float(setup_ms),
        provenance={
            "dataset_id": PICMUS_FILE,
            "dataset_type": "experimental resolution/distortion phantom",
            "source_url": PICMUS_URL,
            "retrieval": "mach.io.utils.cached_download; cached outside the repository",
            "raw_file_size_bytes": data_path.stat().st_size,
            "raw_file_sha256": PICMUS_SHA256,
            "file_integrity": "verified by mach cached_download against expected size and SHA-256",
            "channel_data_shape_transmits_elements_samples_frames": list(channel_array.shape),
            "transmit_count": int(channel_array.shape[0]),
            "element_count": int(channel_data.probe.N),
            "samples_per_transmit": int(channel_array.shape[2]),
            "frame_count": int(channel_array.shape[3]),
            "transmit_angles_deg_min_max": [min(tx_angles_deg), max(tx_angles_deg)],
            "sampling_frequency_hz": float(channel_data.sampling_frequency),
            "modulation_frequency_hz": float(channel_data.modulation_frequency),
            "sound_speed_m_s": float(channel_data.sound_speed),
            "scan_grid_shape_lateral_depth": list(
                (int(scan.x_axis.size), int(scan.z_axis.size))
            ),
            "uff_authorship_metadata": authorship or None,
            "license_note": (
                "The raw UFF file is downloaded and cached, not redistributed by "
                "this repository. Follow the dataset's embedded attribution and "
                "published reuse terms."
            ),
        },
    )


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


def benchmark_picmus(
    *,
    mode: str,
    runs: int,
    warmup: int,
    deadline_ms: float | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure single-frame, 75-angle coherent compounding latency."""
    if mode not in ("numpy", "cupy"):
        raise ValueError(f"Unknown array mode: {mode}")
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup nonnegative")

    from mach import experimental

    workload = load_picmus_workload()
    beamform_args = dict(workload.beamform_args)
    cp = None
    if mode == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy is required for `--mode cupy`; install the CUDA extra on "
                "the NVIDIA host."
            ) from exc

    input_upload_ms = 0.0
    if cp is not None:
        upload_start_ns = time.perf_counter_ns()
        for key in ARRAY_ARGUMENTS:
            beamform_args[key] = cp.asarray(beamform_args[key])
        cp.cuda.Device().synchronize()
        input_upload_ms = (time.perf_counter_ns() - upload_start_ns) / 1_000_000

    event_cp = cp
    if event_cp is None:
        try:
            import cupy as event_cp
        except ImportError:
            event_cp = None

    def run_once() -> tuple[Any, float, float | None]:
        event_start = event_end = None
        if event_cp is not None:
            event_start = event_cp.cuda.Event()
            event_end = event_cp.cuda.Event()
            event_start.record()
        start_ns = time.perf_counter_ns()
        result = experimental.beamform(**beamform_args)
        if event_end is not None:
            event_end.record()
            event_cp.cuda.Device().synchronize()
            device_ms = float(event_start.elapsed_time(event_end))
        else:
            device_ms = None
        wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        return result, float(wall_ms), device_ms

    for _ in range(warmup):
        run_once()

    wall_latencies: list[float] = []
    device_latencies: list[float] = []
    latest_result: Any = None
    for _ in range(runs):
        latest_result, wall_ms, device_ms = run_once()
        wall_latencies.append(wall_ms)
        if device_ms is not None:
            device_latencies.append(device_ms)

    output_transfer_ms = 0.0
    output_start_ns = time.perf_counter_ns()
    if cp is not None and isinstance(latest_result, cp.ndarray):
        result_host = cp.asnumpy(latest_result)
    else:
        result_host = np.asarray(latest_result)
    if cp is not None:
        cp.cuda.Device().synchronize()
    output_transfer_ms = (time.perf_counter_ns() - output_start_ns) / 1_000_000
    bmode_image = np.abs(result_host).reshape(workload.grid_shape).astype(np.float32)

    deadline_misses = (
        sum(latency > deadline_ms for latency in wall_latencies)
        if deadline_ms is not None
        else None
    )
    report: dict[str, Any] = {
        "dataset": "PICMUS experimental resolution/distortion phantom",
        "data_provenance": workload.provenance,
        "runtime_provenance": None,
        "array_mode": mode,
        "scan_grid_shape_lateral_depth": list(workload.grid_shape),
        "dataset_load_and_setup_ms_once": workload.setup_ms,
        "input_upload_ms_once": float(input_upload_ms),
        "output_bmode_transfer_ms_last_run": float(output_transfer_ms),
        "warmup_runs": warmup,
        "measurement_runs": runs,
        "deadline_ms": deadline_ms,
        "deadline_misses_compounded_image": deadline_misses,
        "deadline_miss_rate": (
            deadline_misses / runs if deadline_misses is not None else None
        ),
        "coherent_compounding_wall_ms": _distribution(wall_latencies),
        "beamform_device_event_ms": _distribution(device_latencies),
        "mean_bmode_magnitude": float(np.mean(bmode_image)),
        "notes": [
            "Each measurement reconstructs one pre-acquired image by coherently compounding all transmit angles.",
            "PICMUS has one frame in this file; these are reconstruction timings, not a streaming fUSI deadline or acquisition-to-output latency.",
            "Dataset download, UFF parsing, setup, and final output transfer are reported separately from per-run wall latency.",
            "In cupy mode, input and geometry arrays are uploaded once and reused; in numpy mode mach receives host arrays on each call.",
        ],
    }
    # Import locally to keep the provenance helper optional for library users.
    from .provenance import runtime_provenance

    report["runtime_provenance"] = runtime_provenance()
    return report, bmode_image
