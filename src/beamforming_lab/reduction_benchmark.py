from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from .beamformer import prepare_geometry
from .provenance import runtime_provenance
from .workload import Workload, load_pymust_doppler_workload


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@contextmanager
def _nvtx_range(cp: Any, name: str, enabled: bool) -> Iterator[None]:
    if enabled:
        cp.cuda.nvtx.RangePush(name)
    try:
        yield
    finally:
        if enabled:
            cp.cuda.nvtx.RangePop()


def _numpy_power_map(beamformed: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(beamformed) ** 2, axis=-1)


def _cupy_power_map(cp: Any, beamformed: Any) -> Any:
    return cp.sum(cp.abs(beamformed) ** 2, axis=-1)


def _beamform_reference(
    workload: Workload,
    geometry: Any,
    cp: Any,
    *,
    batch_size: int,
    nvtx: bool,
) -> tuple[Any, np.ndarray]:
    from mach.kernel import beamform

    frame_count = workload.iq_elements_samples_frames.shape[2]
    with _nvtx_range(cp, "setup/iq_upload", nvtx):
        channel_data = cp.asarray(workload.iq_elements_samples_frames)
        cp.cuda.Device().synchronize()

    beamformed_chunks = []
    with _nvtx_range(cp, "setup/beamform_reference", nvtx):
        for first_frame in range(0, frame_count, batch_size):
            last_frame = min(first_frame + batch_size, frame_count)
            chunk = cp.ascontiguousarray(
                channel_data[:, :, first_frame:last_frame]
            )
            beamformed_chunks.append(
                beamform(
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
            )
        if len(beamformed_chunks) == 1:
            beamformed_device = cp.ascontiguousarray(beamformed_chunks[0])
        else:
            beamformed_device = cp.ascontiguousarray(
                cp.concatenate(beamformed_chunks, axis=-1)
            )
        cp.cuda.Device().synchronize()

    if beamformed_device.shape[-1] != frame_count:
        raise ValueError(
            "Beamformer output's last axis must contain the full slow-time ensemble; "
            f"got shape {beamformed_device.shape} for {frame_count} frames."
        )
    if beamformed_device.dtype != cp.complex64:
        raise ValueError(
            "The reduction comparison expects complex64 beamformer output; "
            f"got {beamformed_device.dtype}."
        )

    with _nvtx_range(cp, "setup/reference_d2h", nvtx):
        beamformed_host = cp.asnumpy(beamformed_device)
    return beamformed_device, beamformed_host


def _measure_numpy_batches(
    beamformed: np.ndarray,
    *,
    batches: int,
    repeats_per_batch: int,
    cp: Any,
    nvtx: bool,
    stage_name: str,
) -> tuple[list[float], np.ndarray]:
    samples_ms: list[float] = []
    latest_map: np.ndarray | None = None
    for _ in range(batches):
        start_ns = time.perf_counter_ns()
        with _nvtx_range(cp, stage_name, nvtx):
            for _ in range(repeats_per_batch):
                latest_map = _numpy_power_map(beamformed)
        samples_ms.append(
            (time.perf_counter_ns() - start_ns) / 1_000_000 / repeats_per_batch
        )
    assert latest_map is not None
    return samples_ms, latest_map


def _measure_cupy_batches(
    cp: Any,
    beamformed: Any,
    *,
    batches: int,
    repeats_per_batch: int,
    nvtx: bool,
    stage_name: str,
) -> tuple[list[float], list[float], Any]:
    wall_samples_ms: list[float] = []
    event_samples_ms: list[float] = []
    latest_map = None
    for _ in range(batches):
        cp.cuda.Device().synchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        wall_start_ns = time.perf_counter_ns()
        start_event.record()
        with _nvtx_range(cp, stage_name, nvtx):
            for _ in range(repeats_per_batch):
                latest_map = _cupy_power_map(cp, beamformed)
        end_event.record()
        cp.cuda.Device().synchronize()
        wall_samples_ms.append(
            (time.perf_counter_ns() - wall_start_ns)
            / 1_000_000
            / repeats_per_batch
        )
        event_samples_ms.append(
            float(cp.cuda.get_elapsed_time(start_event, end_event))
            / repeats_per_batch
        )
    return wall_samples_ms, event_samples_ms, latest_map


def benchmark_power_doppler_reduction(
    *,
    grid_size: int = 251,
    batch_size: int | None = None,
    runs: int = 20,
    warmup: int = 5,
    batch_repeats: int = 5,
    nvtx: bool = False,
) -> dict[str, Any]:
    """Compare CPU/GPU reductions of the same beamformed ensemble.

    Timed phases cover reduction with inputs already resident on each processor,
    then the practical post-beamforming paths from a GPU-resident ensemble:
    full-ensemble D2H plus CPU reduction versus GPU reduction plus map D2H.
    """
    if runs < 1 or warmup < 0 or batch_repeats < 1:
        raise ValueError("runs and batch_repeats must be positive; warmup nonnegative")

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for `beamform-lab benchmark-reduction`; run this "
            "benchmark on the Colab T4 or another supported NVIDIA GPU host."
        ) from exc

    workload = load_pymust_doppler_workload(grid_size=grid_size)
    frame_count = workload.iq_elements_samples_frames.shape[2]
    effective_batch_size = frame_count if batch_size is None else batch_size
    if effective_batch_size < 1 or effective_batch_size > frame_count:
        raise ValueError(f"batch_size must be in [1, {frame_count}]")

    with _nvtx_range(cp, "setup/geometry_upload", nvtx):
        geometry = prepare_geometry(workload, "cupy")
    beamformed_device, beamformed_host = _beamform_reference(
        workload,
        geometry,
        cp,
        batch_size=effective_batch_size,
        nvtx=nvtx,
    )

    grid_shape = (len(workload.x_m), len(workload.z_m))
    map_bytes = int(np.prod(grid_shape)) * np.dtype(np.float32).itemsize
    beamformed_bytes = int(beamformed_device.nbytes)

    latest_numpy_map = None
    latest_cupy_map = None
    for _ in range(warmup):
        with _nvtx_range(cp, "warmup/compute_only/numpy", nvtx):
            latest_numpy_map = _numpy_power_map(beamformed_host)
        with _nvtx_range(cp, "warmup/compute_only/cupy", nvtx):
            latest_cupy_map = _cupy_power_map(cp, beamformed_device)
    cp.cuda.Device().synchronize()

    numpy_compute_ms: list[float] = []
    cupy_compute_wall_ms: list[float] = []
    cupy_compute_event_ms: list[float] = []
    for batch_index in range(runs):
        order = ("numpy", "cupy") if batch_index % 2 == 0 else ("cupy", "numpy")
        for mode in order:
            if mode == "numpy":
                measured, latest_numpy_map = _measure_numpy_batches(
                    beamformed_host,
                    batches=1,
                    repeats_per_batch=batch_repeats,
                    cp=cp,
                    nvtx=nvtx,
                    stage_name="measure/compute_only/numpy",
                )
                numpy_compute_ms.extend(measured)
            else:
                wall, device, latest_cupy_map = _measure_cupy_batches(
                    cp,
                    beamformed_device,
                    batches=1,
                    repeats_per_batch=batch_repeats,
                    nvtx=nvtx,
                    stage_name="measure/compute_only/cupy",
                )
                cupy_compute_wall_ms.extend(wall)
                cupy_compute_event_ms.extend(device)

    if latest_numpy_map is None or latest_cupy_map is None:
        raise RuntimeError("Reduction warmup and measurement did not produce maps.")

    numpy_map = latest_numpy_map.reshape(grid_shape)
    cupy_map = cp.asnumpy(latest_cupy_map).reshape(grid_shape)
    difference = numpy_map.astype(np.float64) - cupy_map.astype(np.float64)
    numpy_norm = float(np.linalg.norm(numpy_map.astype(np.float64)))
    shared_peak = max(float(np.max(numpy_map)), float(np.max(cupy_map)))
    correctness = {
        "allclose_rtol_2e-5_atol_1e-7": bool(
            np.allclose(numpy_map, cupy_map, rtol=2e-5, atol=1e-7)
        ),
        "normalized_l2_error": float(np.linalg.norm(difference) / numpy_norm)
        if numpy_norm
        else 0.0,
        "maximum_absolute_difference_over_shared_peak": (
            float(np.max(np.abs(difference)) / shared_peak) if shared_peak else 0.0
        ),
    }
    if not correctness["allclose_rtol_2e-5_atol_1e-7"]:
        raise RuntimeError(f"CPU and GPU reduction outputs disagree: {correctness}")

    numpy_post_ms: list[float] = []
    cupy_post_ms: list[float] = []
    latest_numpy_post_map = None
    latest_cupy_post_map = None
    latest_numpy_feature = None
    latest_cupy_feature = None
    for _ in range(warmup):
        with _nvtx_range(cp, "warmup/post_beamforming/numpy_d2h_reduce", nvtx):
            warm_host = cp.asnumpy(beamformed_device)
            warm_numpy_map = _numpy_power_map(warm_host)
            float(np.mean(warm_numpy_map))
        with _nvtx_range(cp, "warmup/post_beamforming/cupy_reduce_d2h", nvtx):
            warm_cupy_map = _cupy_power_map(cp, beamformed_device)
            float(cp.mean(warm_cupy_map).get())
            cp.asnumpy(warm_cupy_map)
    cp.cuda.Device().synchronize()

    for batch_index in range(runs):
        order = ("numpy", "cupy") if batch_index % 2 == 0 else ("cupy", "numpy")
        for mode in order:
            if mode == "numpy":
                start_ns = time.perf_counter_ns()
                with _nvtx_range(cp, "measure/post_beamform/numpy_d2h_reduce", nvtx):
                    for _ in range(batch_repeats):
                        beamformed_on_host = cp.asnumpy(beamformed_device)
                        latest_numpy_post_map = _numpy_power_map(beamformed_on_host)
                        latest_numpy_feature = float(np.mean(latest_numpy_post_map))
                numpy_post_ms.append(
                    (time.perf_counter_ns() - start_ns)
                    / 1_000_000
                    / batch_repeats
                )
            else:
                cp.cuda.Device().synchronize()
                start_ns = time.perf_counter_ns()
                with _nvtx_range(cp, "measure/post_beamform/cupy_reduce_d2h", nvtx):
                    for _ in range(batch_repeats):
                        gpu_map = _cupy_power_map(cp, beamformed_device)
                        latest_cupy_feature = float(cp.mean(gpu_map).get())
                        latest_cupy_post_map = cp.asnumpy(gpu_map)
                cupy_post_ms.append(
                    (time.perf_counter_ns() - start_ns)
                    / 1_000_000
                    / batch_repeats
                )

    if latest_numpy_post_map is None or latest_cupy_post_map is None:
        raise RuntimeError("Post-beamforming measurements did not produce maps.")
    post_numpy_map = latest_numpy_post_map.reshape(grid_shape)
    post_cupy_map = latest_cupy_post_map.reshape(grid_shape)
    post_maps_close = bool(
        np.allclose(post_numpy_map, post_cupy_map, rtol=2e-5, atol=1e-7)
    )
    post_feature_close = bool(
        np.isclose(latest_numpy_feature, latest_cupy_feature, rtol=2e-5, atol=1e-7)
    )

    return {
        "dataset": "PyMUST rotating-disk Doppler raw-RF phantom",
        "data_provenance": workload.data_provenance,
        "runtime_provenance": runtime_provenance(),
        "reference_beamformed_ensemble": {
            "generated_once_with": "mach CUDA beamformer using CuPy-resident inputs",
            "shape": list(beamformed_device.shape),
            "dtype": str(beamformed_device.dtype),
            "bytes": beamformed_bytes,
            "frames": frame_count,
            "map_shape": list(grid_shape),
            "map_bytes": map_bytes,
        },
        "measurement_design": {
            "warmup_calls_per_backend_per_phase": warmup,
            "batches_per_backend_per_phase": runs,
            "calls_per_batch": batch_repeats,
            "calls_per_backend_per_phase": runs * batch_repeats,
            "distribution_sample": (
                "Each reported timing sample is the average of calls_per_batch "
                "reduction calls; percentile distributions cover batches."
            ),
            "nvtx_ranges_enabled": nvtx,
            "timing_scope": (
                "Compute-only timings start with identical beamformed values already "
                "resident in host and device memory. Post-beamforming path timings "
                "start with the reference ensemble on the GPU and include the full "
                "ensemble D2H plus NumPy reduction, or the CuPy reduction plus the "
                "Power Doppler map and scalar-summary D2H. Reference beamforming and "
                "its initial setup are excluded from all reducer timings."
            ),
        },
        "compute_only_ms_per_call": {
            "numpy_cpu_wall": _distribution(numpy_compute_ms),
            "cupy_gpu_event_elapsed": _distribution(cupy_compute_event_ms),
            "cupy_synchronized_host_wall": _distribution(cupy_compute_wall_ms),
        },
        "post_beamforming_to_host_ms_per_call": {
            "numpy_full_ensemble_d2h_plus_reduction_and_mean": _distribution(
                numpy_post_ms
            ),
            "cupy_reduction_plus_map_and_scalar_d2h": _distribution(cupy_post_ms),
        },
        "correctness": {
            **correctness,
            "post_beamforming_maps_close": post_maps_close,
            "post_beamforming_scalar_summary_close": post_feature_close,
        },
        "notes": [
            "Unprofiled JSON timings are the comparison results; Nsight traces are for stage and kernel attribution.",
            "GPU-event elapsed time and synchronized host wall time are separate metrics.",
            "The compute-only phase excludes transfers; the post-beamforming phase includes the transfer each path requires.",
            "The scalar mean is included only in the post-beamforming path to match the production output summary.",
        ],
    }
