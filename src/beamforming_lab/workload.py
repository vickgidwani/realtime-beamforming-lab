from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Workload:
    iq_elements_samples_frames: np.ndarray
    rx_coords_m: np.ndarray
    scan_coords_m: np.ndarray
    tx_wave_arrivals_s: np.ndarray
    x_m: np.ndarray
    z_m: np.ndarray
    params: dict
    rf_to_iq_ms: float
    data_provenance: dict


def convert_rf_to_iq(rf_data: np.ndarray, params: dict) -> np.ndarray:
    """Convert RF to mach's contiguous elements × samples × frames IQ layout."""
    import pymust

    iq_samples_elements_frames = pymust.rf2iq(rf_data, params)
    return np.ascontiguousarray(
        np.transpose(iq_samples_elements_frames, (1, 0, 2)),
        dtype=np.complex64,
    )


def load_pymust_doppler_workload(grid_size: int = 251) -> Workload:
    """Download/cache the small raw-RF phantom and prepare mach inputs."""
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")

    try:
        import pymust
        from mach import wavefront
        from mach.io.must import (
            download_pymust_doppler_data,
            extract_pymust_params,
            linear_probe_positions,
            scan_grid,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the matching GPU extra on a supported Linux/NVIDIA system: "
            "`python -m pip install -e '.[gpu]'` for CUDA 12.x or "
            "`python -m pip install -e '.[gpu-colab]'` for Colab CUDA 13.x."
        ) from exc

    mat_data = download_pymust_doppler_data()
    params = extract_pymust_params(mat_data)
    raw_rf = np.asarray(mat_data["RF"])
    canonical_rf = np.ascontiguousarray(raw_rf)
    fingerprint = hashlib.sha256()
    fingerprint.update(canonical_rf.dtype.str.encode("ascii"))
    fingerprint.update(np.asarray(canonical_rf.shape, dtype="<i8").tobytes())
    fingerprint.update(canonical_rf.tobytes(order="C"))
    rf_data = raw_rf.astype(float)

    conversion_start = time.perf_counter_ns()
    iq_elements_samples_frames = convert_rf_to_iq(rf_data, params)
    rf_to_iq_ms = (time.perf_counter_ns() - conversion_start) / 1_000_000

    rx_coords_m = np.ascontiguousarray(
        linear_probe_positions(params["Nelements"], params["pitch"]),
        dtype=np.float32,
    )
    x_m = np.linspace(-12.5e-3, 12.5e-3, num=grid_size, endpoint=True, dtype=np.float32)
    y_m = np.array([0.0], dtype=np.float32)
    z_m = np.linspace(10e-3, 35e-3, num=grid_size, endpoint=True, dtype=np.float32)
    scan_coords_m = np.ascontiguousarray(scan_grid(x_m, y_m, z_m), dtype=np.float32)
    tx_wave_arrivals_s = np.ascontiguousarray(
        wavefront.plane(
            origin_m=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            points_m=scan_coords_m,
            direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        / params["c"],
        dtype=np.float32,
    )

    return Workload(
        iq_elements_samples_frames=iq_elements_samples_frames,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        x_m=x_m,
        z_m=z_m,
        params=params,
        rf_to_iq_ms=float(rf_to_iq_ms),
        data_provenance={
            "dataset_id": "PyMUST example PWI_disk.mat",
            "source_url": (
                "https://github.com/creatis-ULTIM/PyMUST/blob/"
                "fe73cc3911a51b48b36095c2e0ad29c266532447/"
                "examples/data/PWI_disk.mat"
            ),
            "retrieval": "mach.io.must.download_pymust_doppler_data()",
            "raw_file_size_bytes": 2_085_623,
            "raw_file_sha256": (
                "c349dc1d677c561434fd0e4a74142c4a0a44b7e6ae0a42d446c078a528ef58c1"
            ),
            "recording": (
                "Ultrasound RF from a tissue-mimicking phantom with a rotating disk; "
                "not a brain or fUSI recording."
            ),
            "rf_array_shape": list(raw_rf.shape),
            "rf_array_dtype": str(raw_rf.dtype),
            "rf_array_sha256": fingerprint.hexdigest(),
            "hash_definition": (
                "SHA-256 of dtype string, little-endian int64 shape, and the loaded "
                "RF array bytes in C order; this fingerprints the RF array, not the "
                "original MAT file bytes."
            ),
            "citation": (
                "Garcia, D. MUST: An Open Platform for Ultrasound Research. "
                "See the PyMUST and mach dataset references in DATA_PROVENANCE.md."
            ),
            "license_note": (
                "The raw data is downloaded from its upstream source and is not "
                "redistributed by this repository. Dataset terms are separate from "
                "the repository's MIT code license."
            ),
        },
    )
