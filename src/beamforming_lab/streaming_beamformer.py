"""Scheduled PyMUST replay for the Exercise 5 notebook and command-line tools."""

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
import time
from typing import TYPE_CHECKING

import numpy as np

from .beamformer import prepare_geometry, run_ensemble
from .streaming import (
    BatchStreamSlot,
    StreamSlot,
    enqueue_h2d_copy,
    load_cython_scheduler,
    make_batch_stream_slot,
    make_stream_slot,
    produce_cython_arrivals,
    produce_python_arrivals,
)
from .workload import load_pymust_doppler_workload

if TYPE_CHECKING:
    from .windowed_inference import WindowedInference

CYCLES_PER_TRIAL = 256
GRID_SIZE = 251
POLL_INTERVAL_S = 0.0001
QUEUE_CAPACITY = 8


@dataclass
class PackedBatch:
    host: np.ndarray
    jobs: list[dict]
    stage_start_ns: int
    stage_ms: float


class StreamingBeamformer:
    """Own one prepared phantom workload and its scheduled replay configuration."""

    def __init__(self, grid_size=GRID_SIZE, cycles_per_trial=CYCLES_PER_TRIAL):
        import cupy as cp
        from mach.kernel import beamform

        self.cp = cp
        self.beamform = beamform
        try:
            from cupy.cuda import nvtx
        except ImportError:
            nvtx = None
        self.nvtx = nvtx

        self.workload = load_pymust_doppler_workload(grid_size=grid_size)
        self.geometry = prepare_geometry(self.workload, "cupy")
        self.channel_data = self.workload.iq_elements_samples_frames
        self.frame_count = int(self.channel_data.shape[2])
        self.frame_shape = (*self.channel_data.shape[:2], 1)
        self.frame_bytes = int(np.prod(self.frame_shape) * np.dtype(np.complex64).itemsize)
        self.replay_frames = self.frame_count * cycles_per_trial

    def make_slot(self, default_stream: bool = False, compute_stream=None):
        return make_stream_slot(
            self.cp,
            self.frame_shape,
            self.frame_bytes,
            default_stream=default_stream,
            compute_stream=compute_stream,
        )


    def warmup(self, explicit_stream: bool = False):
        self.cp.cuda.Device().synchronize()
        warm_slot = self.make_slot()
        np.copyto(warm_slot.host, self.channel_data[:, :, :1])
        enqueue_h2d_copy(
            self.cp,
            warm_slot.host,
            warm_slot.device,
            self.frame_bytes,
            warm_slot.copy_stream,
            warm_slot.copy_start,
            warm_slot.copy_done,
        )
        warm_slot.copy_stream.synchronize()
        with warm_slot.compute_stream:
            warmup_kwargs = {}
            if explicit_stream:
                warmup_kwargs = {"synchronize": False, "stream": warm_slot.compute_stream}
            image = self.beamform(
                channel_data=warm_slot.device,
                rx_coords_m=self.geometry.rx_coords_m,
                scan_coords_m=self.geometry.scan_coords_m,
                tx_wave_arrivals_s=self.geometry.tx_wave_arrivals_s,
                f_number=float(self.workload.params["fnumber"]),
                rx_start_s=float(self.workload.params["t0"]),
                sampling_freq_hz=float(self.workload.params["fs"]),
                sound_speed_m_s=float(self.workload.params["c"]),
                modulation_freq_hz=float(self.workload.params["fc"]),
                tukey_alpha=0.0,
                **warmup_kwargs,
            )
            self.cp.sum(self.cp.abs(image) ** 2, axis=-1)
        warm_slot.compute_stream.synchronize()
        del warm_slot


    def run_trial(self, mode: str, input_rate_hz: int, producer_kind: str = "python") -> dict:
        if mode not in {"serialized", "overlapped", "synchronous_default", "async", "async_explicit_stream"}:
            raise ValueError("unsupported submission mode")
        if producer_kind not in {"python", "cython"}:
            raise ValueError("producer_kind must be 'python' or 'cython'")
        if input_rate_hz <= 0:
            raise ValueError("input_rate_hz must be positive")

        slot_count = 1 if mode == "serialized" else 2
        default_stream = mode in {"synchronous_default", "async"}
        explicit_compute_stream = (
            self.cp.cuda.Stream(non_blocking=True) if mode == "async_explicit_stream" else None
        )
        slots = [
            self.make_slot(default_stream=default_stream, compute_stream=explicit_compute_stream)
            for _ in range(slot_count)
        ]
        power_accumulator = self.cp.zeros(len(self.workload.scan_coords_m), dtype=self.cp.float32)
        self.cp.cuda.Device().synchronize()
        work_queue = Queue(maxsize=QUEUE_CAPACITY)
        producer_done = Event()
        arrivals = []
        drops = []
        producer_errors = []
        completed = []
        max_queue_depth = 0
        period_ns = int(round(1e9 / input_rate_hz))
        offered_count = self.replay_frames
        trial_start_ns = time.perf_counter_ns()

        def produce():
            nonlocal max_queue_depth
            try:
                producer = (
                    produce_cython_arrivals
                    if producer_kind == "cython"
                    else produce_python_arrivals
                )
                max_queue_depth = max(
                    max_queue_depth,
                    producer(
                        work_queue,
                        arrivals,
                        drops,
                        offered_count,
                        self.frame_count,
                        trial_start_ns,
                        period_ns,
                    ),
                )
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                producer_done.set()

        def submit(slot: StreamSlot, job: dict):
            stage_start_ns = time.perf_counter_ns()
            np.copyto(slot.host, self.channel_data[:, :, job["frame_index"]:job["frame_index"] + 1])
            slot.host_stage_ms = (time.perf_counter_ns() - stage_start_ns) / 1e6
            slot.job = {**job, "stage_start_ns": stage_start_ns}

            enqueue_h2d_copy(
                self.cp,
                slot.host,
                slot.device,
                self.frame_bytes,
                slot.copy_stream,
                slot.copy_start,
                slot.copy_done,
            )

            if mode == "serialized":
                slot.copy_done.synchronize()
            else:
                slot.compute_stream.wait_event(slot.copy_done)

            with slot.compute_stream:
                slot.compute_start.record(slot.compute_stream)
                beamform_kwargs = {
                    "channel_data": slot.device,
                    "rx_coords_m": self.geometry.rx_coords_m,
                    "scan_coords_m": self.geometry.scan_coords_m,
                    "tx_wave_arrivals_s": self.geometry.tx_wave_arrivals_s,
                    "f_number": float(self.workload.params["fnumber"]),
                    "rx_start_s": float(self.workload.params["t0"]),
                    "sampling_freq_hz": float(self.workload.params["fs"]),
                    "sound_speed_m_s": float(self.workload.params["c"]),
                    "modulation_freq_hz": float(self.workload.params["fc"]),
                    "tukey_alpha": 0.0,
                }
                if mode in {"async", "async_explicit_stream"}:
                    beamform_kwargs["synchronize"] = False
                if mode == "async_explicit_stream":
                    beamform_kwargs["stream"] = slot.compute_stream
                image = self.beamform(**beamform_kwargs)
                frame_power = self.cp.sum(self.cp.abs(image) ** 2, axis=-1)
                self.cp.add(power_accumulator, frame_power, out=power_accumulator)
                # Keep temporaries alive until the asynchronous reduction finishes.
                slot.pending_refs = (image, frame_power)
                slot.compute_done.record(slot.compute_stream)

            if mode == "serialized":
                slot.compute_done.synchronize()
                finish(slot)

        def finish(slot: StreamSlot):
            completed_ns = time.perf_counter_ns()
            job = slot.job
            completed.append({
                **job,
                "completed_ns": completed_ns,
                "queue_wait_ms": (job["stage_start_ns"] - job["arrived_ns"]) / 1e6,
                "host_latency_ms": (completed_ns - job["arrived_ns"]) / 1e6,
                "producer_lateness_ms": (job["arrived_ns"] - job["scheduled_ns"]) / 1e6,
                "h2d_ms": float(self.cp.cuda.get_elapsed_time(slot.copy_start, slot.copy_done)),
                "gpu_beamform_power_ms": float(self.cp.cuda.get_elapsed_time(slot.compute_start, slot.compute_done)),
                "host_stage_copy_ms": slot.host_stage_ms,
                "deadline_miss": completed_ns > job["scheduled_ns"] + period_ns,
            })
            slot.job = None
            slot.pending_refs = ()

        _nvtx_pushed = False
        if self.nvtx is not None:
            self.nvtx.RangePush(f"exercise5/{producer_kind}/{mode}/{input_rate_hz}Hz")
            _nvtx_pushed = True
        producer = Thread(target=produce, name=f"exercise5-{producer_kind}-producer", daemon=True)
        producer.start()
        try:
            while True:
                progressed = False
                for slot in slots:
                    if slot.job is not None and slot.compute_done.done:
                        finish(slot)
                        progressed = True
                free_slot = next((slot for slot in slots if slot.job is None), None)
                if free_slot is not None:
                    try:
                        job = work_queue.get_nowait()
                    except Empty:
                        job = None
                    if job is not None:
                        job["queue_depth_after_dequeue"] = work_queue.qsize()
                        submit(free_slot, job)
                        progressed = True
                max_queue_depth = max(max_queue_depth, work_queue.qsize())
                if producer_done.is_set() and work_queue.empty() and all(slot.job is None for slot in slots):
                    break
                if not progressed:
                    time.sleep(POLL_INTERVAL_S)
        finally:
            producer.join()
            for slot in slots:
                if slot.job is not None:
                    slot.compute_done.synchronize()
                    finish(slot)
            if _nvtx_pushed:
                self.nvtx.RangePop()

        if producer_errors:
            raise RuntimeError(f"{producer_kind} producer failed") from producer_errors[0]

        self.cp.cuda.Device().synchronize()
        power_map = self.cp.asnumpy(power_accumulator).reshape(len(self.workload.x_m), len(self.workload.z_m))
        trial_end_ns = time.perf_counter_ns()
        latencies = np.asarray([record["host_latency_ms"] for record in completed], dtype=np.float64)
        producer_lateness = np.asarray([(record["arrived_ns"] - record["scheduled_ns"]) / 1e6 for record in arrivals], dtype=np.float64)
        h2d_times = np.asarray([record["h2d_ms"] for record in completed], dtype=np.float64)
        gpu_times = np.asarray([record["gpu_beamform_power_ms"] for record in completed], dtype=np.float64)
        accepted = len(completed)
        first_arrival_ns = arrivals[0]["arrived_ns"] if arrivals else trial_start_ns
        last_arrival_ns = arrivals[-1]["arrived_ns"] if arrivals else trial_start_ns
        actual_producer_rate = ((len(arrivals) - 1) * 1e9 / (last_arrival_ns - first_arrival_ns)
                                if len(arrivals) > 1 and last_arrival_ns > first_arrival_ns else 0.0)

        return {
            "mode": mode,
            "producer_kind": producer_kind,
            "requested_rate_hz": input_rate_hz,
            "actual_producer_rate_hz": float(actual_producer_rate),
            "offered_frames": offered_count,
            "accepted_frames": accepted,
            "dropped_frames": len(drops),
            "deadline_misses": int(sum(record["deadline_miss"] for record in completed)),
            "deadline_miss_fraction": float(np.mean([record["deadline_miss"] for record in completed])) if accepted else None,
            "max_waiting_queue_depth": int(max_queue_depth),
            "trial_wall_s": (trial_end_ns - trial_start_ns) / 1e9,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if accepted else None,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if accepted else None,
            "latency_p99_ms": float(np.percentile(latencies, 99)) if accepted else None,
            "latency_max_ms": float(np.max(latencies)) if accepted else None,
            "producer_lateness_p50_ms": float(np.percentile(producer_lateness, 50)) if len(producer_lateness) else None,
            "producer_lateness_p95_ms": float(np.percentile(producer_lateness, 95)) if len(producer_lateness) else None,
            "h2d_event_p50_ms": float(np.percentile(h2d_times, 50)) if accepted else None,
            "gpu_beamform_power_p50_ms": float(np.percentile(gpu_times, 50)) if accepted else None,
            "power_map": power_map,
            "frame_records": completed,
            "drops": drops,
        }


    def run_batched_trial(
        self,
        batch_size: int,
        input_rate_hz: int = 10_000,
        queue_capacity: int = QUEUE_CAPACITY,
        mode: str = "overlapped",
        *,
        reference: np.ndarray | None = None,
        trace_stages: bool = False,
        packer: str = "strided",
        concurrent_staging: bool = False,
        reuse_beamform_output: bool = False,
        prepare_mach: bool = False,
        trace_mach: bool = False,
        window_inference: "WindowedInference | None" = None,
        omit_sequence: int = -1,
    ) -> dict:
        if mode not in {"overlapped", "synchronous_default", "async", "async_explicit_stream"}:
            raise ValueError("unsupported submission mode")
        if batch_size < 1 or batch_size > self.frame_count:
            raise ValueError("batch_size must be between 1 and the recorded frame count")
        if input_rate_hz <= 0:
            raise ValueError("input_rate_hz must be positive")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")

        if window_inference is not None:
            if mode != "async_explicit_stream" or not concurrent_staging:
                raise ValueError("windowed inference requires explicit-stream concurrent staging")
            if self.replay_frames % window_inference.window_frames:
                raise ValueError("replay must contain a whole number of windows")
        if not -1 <= omit_sequence < self.replay_frames:
            raise ValueError("omit_sequence is outside the replay")

        if reuse_beamform_output and mode != "async_explicit_stream":
            raise ValueError("output reuse requires the explicit compute stream")
        if prepare_mach and not reuse_beamform_output:
            raise ValueError("prepared Mach launches require an explicit reusable output")
        if prepare_mach:
            from mach.kernel import prepare_beamform
        if trace_mach and not trace_stages:
            raise ValueError("Mach tracing requires stage tracing")
        if packer not in {"strided", "one_pass"}:
            raise ValueError("unsupported host packer")
        pack_iq_batch = load_cython_scheduler().pack_iq_batch if packer == "one_pass" else None
        if pack_iq_batch is not None:
            source_words = self.channel_data.view(np.uint64).reshape(-1, self.frame_count)

        nvtx = self.nvtx if trace_stages else None
        if trace_stages and (nvtx is None or not nvtx.available):
            raise RuntimeError("Detailed stage tracing requires CuPy with NVTX support")
        wait_reason = None

        def set_wait(reason: str | None) -> None:
            nonlocal wait_reason
            if reason == wait_reason:
                return
            if wait_reason is not None:
                nvtx.RangePop()
            if reason is not None:
                nvtx.RangePush(f"exercise5/wait/{reason}")
            wait_reason = reason

        def batch_label(jobs: list[dict]) -> str:
            return f"seq={jobs[0]['sequence']}-{jobs[-1]['sequence']}/size={len(jobs)}"

        period_ns = int(round(1e9 / input_rate_hz))
        max_fill_wait_ns = batch_size * period_ns
        offered_count = self.replay_frames
        # One compute stream serializes accumulation into the shared power map.
        # The default-stream modes use the NULL stream; other paths use a non-blocking stream.
        if mode in {"synchronous_default", "async"}:
            compute_stream = self.cp.cuda.Stream.null
        else:
            compute_stream = self.cp.cuda.Stream(non_blocking=True)
        slots = [
            make_batch_stream_slot(self.cp, batch_size, self.frame_shape, self.frame_bytes, compute_stream)
            for _ in range(2)
        ]

        def beamform_arguments(slot: BatchStreamSlot) -> dict:
            beamform_kwargs = {
                "channel_data": slot.device,
                "rx_coords_m": self.geometry.rx_coords_m,
                "scan_coords_m": self.geometry.scan_coords_m,
                "tx_wave_arrivals_s": self.geometry.tx_wave_arrivals_s,
                "f_number": float(self.workload.params["fnumber"]),
                "rx_start_s": float(self.workload.params["t0"]),
                "sampling_freq_hz": float(self.workload.params["fs"]),
                "sound_speed_m_s": float(self.workload.params["c"]),
                "modulation_freq_hz": float(self.workload.params["fc"]),
                "tukey_alpha": 0.0,
            }
            if mode in {"async", "async_explicit_stream"}:
                beamform_kwargs["synchronize"] = False
            if mode == "async_explicit_stream":
                beamform_kwargs["stream"] = slot.compute_stream
            if trace_mach:
                beamform_kwargs["trace_submission"] = True
            return beamform_kwargs

        prepared_arguments = {}
        prepared_launches = {}
        if reuse_beamform_output:
            for slot in slots:
                arguments = beamform_arguments(slot)
                arguments["out"] = self.cp.empty(
                    (len(self.workload.scan_coords_m), batch_size), dtype=self.cp.complex64
                )
                prepared_arguments[id(slot)] = arguments
                if prepare_mach:
                    prepared_launches[id(slot)] = prepare_beamform(**arguments)

        # The slots retain the allocation owners. In concurrent mode their host
        # arrays circulate independently from their device buffers.
        free_hosts = Queue(maxsize=len(slots))
        ready_batches = Queue(maxsize=len(slots))
        copying_hosts = {}
        staging_done = Event()
        stop_staging = Event()
        staging_errors = []
        if concurrent_staging:
            for slot in slots:
                free_hosts.put_nowait(slot.host)

        power_accumulator = None
        if window_inference is None:
            power_accumulator = self.cp.zeros(len(self.workload.scan_coords_m), dtype=self.cp.float32)
        self.cp.cuda.Device().synchronize()
        work_queue = Queue(maxsize=queue_capacity)
        producer_done = Event()
        arrivals, drops, producer_errors = [], [], []
        max_queue_depth = [0]
        completed, batch_records = [], []
        max_queue_depth_seen = 0
        trial_start_ns = time.perf_counter_ns()

        def produce():
            try:
                max_queue_depth[0] = produce_cython_arrivals(
                    work_queue,
                    arrivals,
                    drops,
                    offered_count,
                    self.frame_count,
                    trial_start_ns,
                    period_ns,
                    omit_sequence,
                )
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                producer_done.set()

        def pack(host: np.ndarray, jobs: list[dict]) -> PackedBatch:
            stage_start_ns = time.perf_counter_ns()
            label = batch_label(jobs) if nvtx is not None else None
            if nvtx is not None:
                nvtx.RangePush(f"exercise5/host_staging/{label}")
            try:
                if pack_iq_batch is None:
                    for index, job in enumerate(jobs):
                        np.copyto(host[:, :, index], self.channel_data[:, :, job["frame_index"]])
                    if len(jobs) < batch_size:
                        host[:, :, len(jobs):] = 0
                else:
                    indices = np.fromiter(
                        (job["frame_index"] for job in jobs), dtype=np.intp, count=len(jobs)
                    )
                    destination_words = host.view(np.uint64).reshape(-1, batch_size)
                    pack_iq_batch(source_words, destination_words, indices)
            finally:
                if nvtx is not None:
                    nvtx.RangePop()
            return PackedBatch(host, jobs, stage_start_ns, (time.perf_counter_ns() - stage_start_ns) / 1e6)

        deferred_job = None

        def batch_is_ready(pending: list[dict]) -> bool:
            nonlocal deferred_job
            while len(pending) < batch_size:
                try:
                    job = deferred_job if deferred_job is not None else work_queue.get_nowait()
                except Empty:
                    break
                deferred_job = None
                if window_inference is not None:
                    window_frames = window_inference.window_frames
                    if pending and job["sequence"] // window_frames != pending[0]["sequence"] // window_frames:
                        deferred_job = job
                        return True
                    if (job["sequence"] + 1) % window_frames == 0:
                        pending.append(job)
                        return True
                pending.append(job)
            flush_partial = bool(pending) and (
                producer_done.is_set() and work_queue.empty()
                or time.perf_counter_ns() - pending[0]["arrived_ns"] >= max_fill_wait_ns
            )
            return len(pending) == batch_size or flush_partial

        def stage_batches():
            pending = []
            host = None
            try:
                while not stop_staging.is_set():
                    if producer_done.is_set() and work_queue.empty() and not pending and deferred_job is None:
                        return
                    if host is None:
                        try:
                            host = free_hosts.get_nowait()
                        except Empty:
                            time.sleep(POLL_INTERVAL_S)
                            continue
                    if not batch_is_ready(pending):
                        time.sleep(POLL_INTERVAL_S)
                        continue
                    if nvtx is not None:
                        nvtx.Mark(f"exercise5/batch_ready/{batch_label(pending)}")
                    packed = pack(host, pending)
                    if nvtx is not None:
                        nvtx.Mark(f"exercise5/packed_ready/{batch_label(pending)}")
                    ready_batches.put_nowait(packed)
                    host, pending = None, []
            except BaseException as exc:
                staging_errors.append(exc)
            finally:
                staging_done.set()

        def submit(slot: BatchStreamSlot, packed: PackedBatch):
            consumed_ns = time.perf_counter_ns()
            jobs = packed.jobs
            label = batch_label(jobs) if nvtx is not None else None
            slot.stage_start_ns = packed.stage_start_ns
            slot.host_stage_ms = packed.stage_ms
            slot.jobs = jobs
            slot.batch_fill_ms = (jobs[-1]["arrived_ns"] - jobs[0]["arrived_ns"]) / 1e6
            slot.batch_span_slots = jobs[-1]["sequence"] - jobs[0]["sequence"] + 1

            if nvtx is not None:
                nvtx.RangePush(f"exercise5/batch_submit/{label}")
            try:
                enqueue_h2d_copy(
                    self.cp,
                    packed.host,
                    slot.device,
                    self.frame_bytes * batch_size,
                    slot.copy_stream,
                    slot.copy_start,
                    slot.copy_done,
                )

                if concurrent_staging:
                    copying_hosts[id(slot)] = packed.host
                if nvtx is not None:
                    nvtx.RangePush(f"exercise5/dispatch_prepare/{label}")
                try:
                    if reuse_beamform_output:
                        beamform_kwargs = prepared_arguments[id(slot)]
                        out = beamform_kwargs["out"]
                        # Mach atomically accumulates into out. Clear it on the
                        # compute stream before that stream waits for this input.
                        self.cp.cuda.runtime.memsetAsync(
                            out.data.ptr, 0, out.nbytes, slot.compute_stream.ptr
                        )
                    else:
                        beamform_kwargs = None
                    slot.compute_stream.wait_event(slot.copy_done)
                    slot.compute_start.record(slot.compute_stream)
                    if beamform_kwargs is None:
                        beamform_kwargs = beamform_arguments(slot)
                finally:
                    if nvtx is not None:
                        nvtx.RangePop()
                with slot.compute_stream:
                    if nvtx is not None:
                        nvtx.RangePush(f"exercise5/mach_wrapper/{label}")
                    try:
                        if prepare_mach:
                            image = prepared_launches[id(slot)]()
                        else:
                            image = self.beamform(**beamform_kwargs)
                    finally:
                        if nvtx is not None:
                            nvtx.RangePop()
                    batch_power = self.cp.sum(self.cp.abs(image) ** 2, axis=-1)
                    if window_inference is None:
                        self.cp.add(power_accumulator, batch_power, out=power_accumulator)
                    else:
                        window_inference.add_batch(batch_power, jobs, consumed_ns, slot.compute_stream)
                    # Retain asynchronous outputs until the completion event has fired.
                    slot.pending_refs = (image, batch_power)
                    slot.compute_done.record(slot.compute_stream)
            finally:
                if nvtx is not None:
                    nvtx.RangePop()

        def finish(slot: BatchStreamSlot):
            if nvtx is not None:
                set_wait(None)
            completed_ns = time.perf_counter_ns()
            jobs = slot.jobs
            span_slots = slot.batch_span_slots
            deadline_ns = jobs[-1]["scheduled_ns"] + span_slots * period_ns
            batch_late = completed_ns > deadline_ns
            batch_records.append({
                "batch_size": len(jobs),
                "represented_slots": span_slots,
                "batch_fill_ms": slot.batch_fill_ms,
                "queue_wait_ms": (slot.stage_start_ns - jobs[0]["arrived_ns"]) / 1e6,
                "host_stage_ms": slot.host_stage_ms,
                "dispatch_to_done_ms": (completed_ns - slot.stage_start_ns) / 1e6,
                "ready_to_done_ms": (completed_ns - jobs[-1]["arrived_ns"]) / 1e6,
                "batch_deadline_ms": span_slots * period_ns / 1e6,
                "batch_late": batch_late,
                "h2d_ms": float(self.cp.cuda.get_elapsed_time(slot.copy_start, slot.copy_done)),
                "gpu_ms": float(self.cp.cuda.get_elapsed_time(slot.compute_start, slot.compute_done)),
            })
            for job in jobs:
                completed.append({
                    **job,
                    "completed_ns": completed_ns,
                    "host_latency_ms": (completed_ns - job["arrived_ns"]) / 1e6,
                    "scheduled_age_ms": (completed_ns - job["scheduled_ns"]) / 1e6,
                    "queue_wait_ms": (slot.stage_start_ns - job["arrived_ns"]) / 1e6,
                    "batch_deadline_miss": batch_late,
                    "strict_frame_deadline_miss": completed_ns > job["scheduled_ns"] + period_ns,
                })
            slot.jobs = None
            slot.pending_refs = ()
            if nvtx is not None:
                nvtx.Mark(f"exercise5/slot_released/{batch_label(jobs)}")

        _nvtx_pushed = False
        if self.nvtx is not None:
            exercise = "exercise7" if window_inference is not None else "exercise5"
            self.nvtx.RangePush(f"{exercise}/cython/{mode}/{input_rate_hz}Hz/batch{batch_size}/queue{queue_capacity}")
            _nvtx_pushed = True
        producer = Thread(target=produce, name=f"exercise5-cython-batch{batch_size}-producer", daemon=True)
        stager = Thread(target=stage_batches, name="exercise5-host-staging", daemon=True) if concurrent_staging else None
        pending: list[dict] = []
        producer.start()
        if stager is not None:
            stager.start()
        try:
            while True:
                progressed = window_inference.poll() if window_inference is not None else False
                for slot in slots:
                    compute_finished = slot.jobs is not None and slot.compute_done.done
                    if id(slot) in copying_hosts and (compute_finished or slot.copy_done.done):
                        free_hosts.put_nowait(copying_hosts.pop(id(slot)))
                        if nvtx is not None:
                            nvtx.Mark(f"exercise5/host_released/{batch_label(slot.jobs)}")
                        progressed = True
                    if compute_finished:
                        finish(slot)
                        progressed = True

                max_queue_depth_seen = max(max_queue_depth_seen, max_queue_depth[0], work_queue.qsize())
                free_slot = next((slot for slot in slots if slot.jobs is None), None)
                if free_slot is not None:
                    if concurrent_staging:
                        try:
                            packed = ready_batches.get_nowait()
                        except Empty:
                            pass
                        else:
                            if nvtx is not None:
                                set_wait(None)
                            submit(free_slot, packed)
                            progressed = True
                    else:
                        previous_count = len(pending)
                        ready = batch_is_ready(pending)
                        if len(pending) != previous_count:
                            if nvtx is not None:
                                set_wait(None)
                            progressed = True
                        if ready:
                            if nvtx is not None:
                                set_wait(None)
                                nvtx.Mark(f"exercise5/batch_ready/{batch_label(pending)}")
                            submit(free_slot, pack(free_slot.host, pending))
                            pending = []
                            progressed = True

                if concurrent_staging:
                    dispatch_done = staging_done.is_set() and ready_batches.empty()
                else:
                    dispatch_done = producer_done.is_set() and work_queue.empty() and not pending
                if dispatch_done and all(slot.jobs is None for slot in slots) and not copying_hosts:
                    break
                if not progressed:
                    if nvtx is not None:
                        if free_slot is None:
                            set_wait("free_slot")
                        elif concurrent_staging:
                            set_wait("packed_batch")
                        elif pending:
                            set_wait("batch_fill")
                        else:
                            set_wait("input")
                    time.sleep(POLL_INTERVAL_S)
            if window_inference is not None:
                window_inference.drain()
        except BaseException:
            if window_inference is not None:
                self.cp.cuda.Device().synchronize()
            raise
        finally:
            if nvtx is not None:
                set_wait(None)
            if stager is not None:
                stop_staging.set()
                stager.join()
            producer.join()
            # On error, a slot may have submitted a copy before recording compute_done.
            # Keep its pinned and device allocations alive until all submitted work ends.
            if any(slot.jobs is not None for slot in slots) or copying_hosts:
                self.cp.cuda.Device().synchronize()
            if _nvtx_pushed:
                self.nvtx.RangePop()

        if staging_errors:
            raise RuntimeError("Host staging worker failed") from staging_errors[0]
        if producer_errors:
            raise RuntimeError("Cython batch producer failed") from producer_errors[0]
        self.cp.cuda.Device().synchronize()
        power_map = None
        if power_accumulator is not None:
            power_map = self.cp.asnumpy(power_accumulator).reshape(len(self.workload.x_m), len(self.workload.z_m))
        trial_end_ns = time.perf_counter_ns()

        accepted = len(completed)
        batch_late_frames = sum(r["batch_deadline_miss"] for r in completed)
        strict_late_frames = sum(r["strict_frame_deadline_miss"] for r in completed)
        producer_lateness_ms = np.asarray([(r["arrived_ns"] - r["scheduled_ns"]) / 1e6 for r in arrivals])
        host_latency_ms = np.asarray([r["host_latency_ms"] for r in completed])
        scheduled_age_ms = np.asarray([r["scheduled_age_ms"] for r in completed])
        batch_gpu_ms = np.asarray([r["gpu_ms"] for r in batch_records])
        batch_h2d_ms = np.asarray([r["h2d_ms"] for r in batch_records])
        batch_fill_ms = np.asarray([r["batch_fill_ms"] for r in batch_records])
        batch_stage_ms = np.asarray([r["host_stage_ms"] for r in batch_records])
        batch_wall_ms = np.asarray([r["dispatch_to_done_ms"] for r in batch_records])
        batch_ready_ms = np.asarray([r["ready_to_done_ms"] for r in batch_records])
        actual_rate_hz = (
            (len(arrivals) - 1) * 1e9 / (arrivals[-1]["arrived_ns"] - arrivals[0]["arrived_ns"])
            if len(arrivals) > 1 and arrivals[-1]["arrived_ns"] > arrivals[0]["arrived_ns"] else 0.0
        )
        image_relative_rmse = None
        if power_map is not None and accepted == offered_count and not drops:
            if reference is None:
                reference = run_ensemble(self.workload, self.geometry, batch_size=1)["power_doppler"]
            normalized_map = power_map / (offered_count / self.frame_count)
            image_relative_rmse = float(
                np.sqrt(np.mean((normalized_map - reference) ** 2))
                / max(np.sqrt(np.mean(reference ** 2)), np.finfo(float).eps)
            )

        return {
            "producer_kind": "cython",
            "mode": mode,
            "batch_size_target": batch_size,
            "queue_capacity_frames": queue_capacity,
            "requested_rate_hz": input_rate_hz,
            "actual_producer_rate_hz": float(actual_rate_hz),
            "offered_frames": offered_count,
            "accepted_frames": accepted,
            "dropped_frames": len(drops),
            "batch_deadline_miss_frames": int(batch_late_frames),
            "strict_frame_deadline_miss_frames": int(strict_late_frames),
            "batch_deadline_failure_share_pct": 100 * (len(drops) + batch_late_frames) / offered_count,
            "strict_frame_failure_share_pct": 100 * (len(drops) + strict_late_frames) / offered_count,
            "max_waiting_queue_depth": int(max_queue_depth_seen),
            "trial_wall_s": (trial_end_ns - trial_start_ns) / 1e9,
            "completed_frames_per_s": accepted / ((trial_end_ns - trial_start_ns) / 1e9),
            "latency_p50_ms": float(np.percentile(host_latency_ms, 50)) if accepted else None,
            "latency_p95_ms": float(np.percentile(host_latency_ms, 95)) if accepted else None,
            "scheduled_age_p95_ms": float(np.percentile(scheduled_age_ms, 95)) if accepted else None,
            "producer_lateness_p95_ms": float(np.percentile(producer_lateness_ms, 95)) if len(arrivals) else None,
            "batch_size_mean": float(np.mean([b["batch_size"] for b in batch_records])) if batch_records else None,
            "batch_fill_p50_ms": float(np.percentile(batch_fill_ms, 50)) if len(batch_fill_ms) else None,
            "host_stage_p50_ms_per_batch": float(np.percentile(batch_stage_ms, 50)) if len(batch_stage_ms) else None,
            "batch_wall_p50_ms": float(np.percentile(batch_wall_ms, 50)) if len(batch_wall_ms) else None,
            "ready_to_done_p50_ms": float(np.percentile(batch_ready_ms, 50)) if len(batch_ready_ms) else None,
            "batch_deadline_p50_ms": float(np.percentile([b["batch_deadline_ms"] for b in batch_records], 50)) if batch_records else None,
            "h2d_event_p50_ms_per_batch": float(np.percentile(batch_h2d_ms, 50)) if len(batch_h2d_ms) else None,
            "gpu_event_p50_ms_per_batch": float(np.percentile(batch_gpu_ms, 50)) if len(batch_gpu_ms) else None,
            "image_relative_rmse": image_relative_rmse,
            "power_map": power_map,
            "batch_records": batch_records,
            "frame_records": completed,
            "drops": drops,
        }
