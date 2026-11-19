"""Reusable scheduling and CUDA stream helpers for live replay experiments.

Experiment-specific run matrices and result analysis stay in the Exercise 5
notebook. This module holds the native arrival producer, pinned staging slots,
and asynchronous H2D copy primitive shared by its frame and batch paths.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Full
from typing import Any

import numpy as np


_CYTHON_SCHEDULER: Any | None = None


def load_cython_scheduler() -> Any:
    """Compile and cache the Linux absolute-deadline arrival producer."""
    global _CYTHON_SCHEDULER
    if _CYTHON_SCHEDULER is not None:
        return _CYTHON_SCHEDULER
    if sys.platform != "linux":
        raise RuntimeError(
            "The Cython scheduler uses Linux clock_nanosleep and requires a Linux host."
        )
    try:
        import Cython  # noqa: F401
        import pyximport
    except ImportError as exc:
        raise RuntimeError(
            "Install Cython before running the native-producer comparison."
        ) from exc

    config = sysconfig.get_config_vars()
    cflags = config.get("CFLAGS", "")
    if "-D_POSIX_C_SOURCE=200809L" not in cflags:
        config["CFLAGS"] = f"{cflags} -D_POSIX_C_SOURCE=200809L".strip()

    source = Path(__file__).with_name("native_replay.pyx").read_bytes()
    build_key = hashlib.sha256(source).hexdigest()[:12]
    build_root = Path(tempfile.gettempdir()) / f"beamforming-native-replay-{build_key}"
    source_dir = build_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "native_replay.pyx"
    if not source_file.exists():
        source_file.write_bytes(source)
    build_dir = build_root / "build"
    pyximport.install(language_level=3, build_dir=str(build_dir))
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    sys.modules.pop("native_replay", None)
    _CYTHON_SCHEDULER = importlib.import_module("native_replay")
    return _CYTHON_SCHEDULER


def produce_cython_arrivals(
    work_queue: Any,
    arrivals: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    offered_count: int,
    frame_count: int,
    trial_start_ns: int,
    period_ns: int,
    omit_sequence: int = -1,
) -> int:
    """Emit absolute-time arrivals through the compiled producer."""
    return int(
        load_cython_scheduler().produce_arrivals(
            work_queue,
            arrivals,
            drops,
            offered_count,
            frame_count,
            trial_start_ns,
            period_ns,
            omit_sequence,
        )
    )


def produce_python_arrivals(
    work_queue: Any,
    arrivals: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    offered_count: int,
    frame_count: int,
    trial_start_ns: int,
    period_ns: int,
) -> int:
    """Python reference producer with the same absolute schedule and drop policy."""
    import time

    max_depth = 0
    for sequence in range(offered_count):
        scheduled_ns = trial_start_ns + sequence * period_ns
        remaining_ns = scheduled_ns - time.perf_counter_ns()
        if remaining_ns > 200_000:
            time.sleep((remaining_ns - 200_000) / 1e9)
        while time.perf_counter_ns() < scheduled_ns:
            pass
        arrived_ns = time.perf_counter_ns()
        job = {
            "sequence": sequence,
            "frame_index": sequence % frame_count,
            "scheduled_ns": scheduled_ns,
            "arrived_ns": arrived_ns,
            "queue_depth_before_enqueue": work_queue.qsize(),
        }
        arrivals.append(job)
        try:
            work_queue.put_nowait(job)
            max_depth = max(max_depth, work_queue.qsize())
        except Full:
            drops.append(
                {
                    **job,
                    "queue_depth_at_drop": work_queue.qsize(),
                    "reason": "drop_newest_queue_full",
                }
            )
    return max_depth


@dataclass
class StreamSlot:
    """One pinned-host/device frame pair with independent copy and compute streams."""

    owner: Any
    host: np.ndarray
    device: Any
    copy_stream: Any
    compute_stream: Any
    copy_start: Any
    copy_done: Any
    compute_start: Any
    compute_done: Any
    job: dict[str, Any] | None = None
    host_stage_ms: float = 0.0
    pending_refs: tuple[Any, ...] = ()


def make_stream_slot(
    cp: Any,
    frame_shape: tuple[int, ...],
    frame_bytes: int,
    *,
    default_stream: bool = False,
    compute_stream: Any | None = None,
) -> StreamSlot:
    """Allocate pinned staging and CUDA events for one replay frame."""
    owner = cp.cuda.alloc_pinned_memory(frame_bytes)
    host = np.frombuffer(owner, dtype=np.complex64, count=int(np.prod(frame_shape))).reshape(
        frame_shape
    )
    device = cp.empty(frame_shape, dtype=cp.complex64)
    copy_stream = cp.cuda.Stream(non_blocking=True)
    if compute_stream is None:
        compute_stream = cp.cuda.Stream.null if default_stream else cp.cuda.Stream(non_blocking=True)
    return StreamSlot(
        owner=owner,
        host=host,
        device=device,
        copy_stream=copy_stream,
        compute_stream=compute_stream,
        copy_start=cp.cuda.Event(),
        copy_done=cp.cuda.Event(),
        compute_start=cp.cuda.Event(),
        compute_done=cp.cuda.Event(),
    )


@dataclass
class BatchStreamSlot:
    """Pinned-host/device staging and events for one fixed-size batch."""

    owner: Any
    host: np.ndarray
    device: Any
    copy_stream: Any
    compute_stream: Any
    copy_start: Any
    copy_done: Any
    compute_start: Any
    compute_done: Any
    jobs: list[dict[str, Any]] | None = None
    stage_start_ns: int = 0
    host_stage_ms: float = 0.0
    batch_fill_ms: float = 0.0
    batch_span_slots: int = 0
    pending_refs: tuple[Any, ...] = ()


def make_batch_stream_slot(
    cp: Any,
    batch_size: int,
    frame_shape: tuple[int, ...],
    frame_bytes: int,
    compute_stream: Any,
) -> BatchStreamSlot:
    """Allocate pinned staging and CUDA events for one batch slot."""
    batch_shape = (*frame_shape[:2], batch_size)
    batch_bytes = frame_bytes * batch_size
    owner = cp.cuda.alloc_pinned_memory(batch_bytes)
    host = np.frombuffer(owner, dtype=np.complex64, count=int(np.prod(batch_shape))).reshape(
        batch_shape
    )
    device = cp.empty(batch_shape, dtype=cp.complex64)
    copy_stream = cp.cuda.Stream(non_blocking=True)
    return BatchStreamSlot(
        owner=owner,
        host=host,
        device=device,
        copy_stream=copy_stream,
        compute_stream=compute_stream,
        copy_start=cp.cuda.Event(),
        copy_done=cp.cuda.Event(),
        compute_start=cp.cuda.Event(),
        compute_done=cp.cuda.Event(),
    )


def enqueue_h2d_copy(
    cp: Any,
    host: np.ndarray,
    device: Any,
    byte_count: int,
    stream: Any,
    start_event: Any,
    done_event: Any,
) -> None:
    """Enqueue a timed host-to-device copy on a non-default CUDA stream."""
    with stream:
        start_event.record(stream)
        cp.cuda.runtime.memcpyAsync(
            int(device.data.ptr),
            int(host.ctypes.data),
            byte_count,
            cp.cuda.runtime.memcpyHostToDevice,
            int(stream.ptr),
        )
        done_event.record(stream)
