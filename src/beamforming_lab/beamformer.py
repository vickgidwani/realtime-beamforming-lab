from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .workload import Workload


@dataclass
class PreparedGeometry:
    mode: str
    rx_coords_m: Any
    scan_coords_m: Any
    tx_wave_arrivals_s: Any
    cupy: Any | None
    geometry_upload_ms: float


def prepare_geometry(workload: Workload, mode: str) -> PreparedGeometry:
    if mode == "numpy":
        return PreparedGeometry(
            mode=mode,
            rx_coords_m=workload.rx_coords_m,
            scan_coords_m=workload.scan_coords_m,
            tx_wave_arrivals_s=workload.tx_wave_arrivals_s,
            cupy=None,
            geometry_upload_ms=0.0,
        )
    if mode != "cupy":
        raise ValueError(f"Unknown array mode: {mode}")

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for `--mode cupy`; install the CUDA extra on the "
            "NVIDIA host."
        ) from exc

    start_ns = time.perf_counter_ns()
    rx_coords_m = cp.asarray(workload.rx_coords_m)
    scan_coords_m = cp.asarray(workload.scan_coords_m)
    tx_wave_arrivals_s = cp.asarray(workload.tx_wave_arrivals_s)
    cp.cuda.Device().synchronize()
    upload_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    return PreparedGeometry(
        mode=mode,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        cupy=cp,
        geometry_upload_ms=float(upload_ms),
    )


def run_ensemble(
    workload: Workload,
    geometry: PreparedGeometry,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Beamform an ensemble, then reduce it to a Power Doppler map."""
    from mach.kernel import beamform

    frame_count = workload.iq_elements_samples_frames.shape[2]
    if batch_size < 1 or batch_size > frame_count:
        raise ValueError(f"batch_size must be in [1, {frame_count}]")

    cp = geometry.cupy
    xp = np if cp is None else cp
    run_start_ns = time.perf_counter_ns()
    upload_ms = 0.0
    if cp is None:
        channel_data = workload.iq_elements_samples_frames
    else:
        upload_start_ns = time.perf_counter_ns()
        channel_data = cp.asarray(workload.iq_elements_samples_frames)
        cp.cuda.Device().synchronize()
        upload_ms = (time.perf_counter_ns() - upload_start_ns) / 1_000_000

    power_doppler = None
    device_event_pairs: list[tuple[Any, Any]] = []
    beamform_host_ms = 0.0
    for first_frame in range(0, frame_count, batch_size):
        last_frame = min(first_frame + batch_size, frame_count)
        chunk = xp.ascontiguousarray(channel_data[:, :, first_frame:last_frame])

        event_start = event_end = None
        if cp is not None:
            event_start = cp.cuda.Event()
            event_end = cp.cuda.Event()
            event_start.record()

        call_start_ns = time.perf_counter_ns()
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
        beamform_host_ms += (time.perf_counter_ns() - call_start_ns) / 1_000_000

        if cp is not None:
            event_end.record()
            device_event_pairs.append((event_start, event_end))

        chunk_power = xp.sum(xp.abs(beamformed) ** 2, axis=-1)
        if power_doppler is None:
            power_doppler = chunk_power
        else:
            power_doppler += chunk_power

    if cp is not None:
        cp.cuda.Device().synchronize()
        device_beamform_ms = sum(
            float(cp.cuda.get_elapsed_time(event_start, event_end))
            for event_start, event_end in device_event_pairs
        )
        output_feature = float(xp.mean(power_doppler).get())
        power_host = cp.asnumpy(power_doppler)
    else:
        device_beamform_ms = None
        output_feature = float(np.mean(power_doppler))
        power_host = np.asarray(power_doppler)

    run_wall_ms = (time.perf_counter_ns() - run_start_ns) / 1_000_000
    return {
        "power_doppler": power_host.reshape(len(workload.x_m), len(workload.z_m)),
        "output_feature": output_feature,
        "input_upload_ms": float(upload_ms),
        "beamform_host_call_ms": float(beamform_host_ms),
        "beamform_device_ms": device_beamform_ms,
        "run_wall_ms": float(run_wall_ms),
    }
