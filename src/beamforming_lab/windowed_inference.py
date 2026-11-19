"""Accumulate complete replay windows and run a single CUDA-bound encoder."""

from pathlib import Path
import time

import cupy as cp
import numpy as np
import onnxruntime as ort

from .beamformed_image_preprocessing import MODEL_INPUT_SHAPE, preprocess_power_map

EMBEDDING_SHAPE = (1, 2048)


class WindowedInference:
    """Own the imaging accumulator and one in-flight inference handoff."""

    def __init__(
        self, model: Path, map_shape: tuple[int, int], scale: float,
        *, window_frames: int = 32, deadline_ms: float = 100, trace: bool = False,
    ):
        self.window_frames = window_frames
        self.deadline_ns = round(deadline_ms * 1e6)
        self.scale = scale
        self.trace = trace
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.accumulator = cp.empty(map_shape, dtype=cp.float32)
        self.handoff = cp.empty_like(self.accumulator)
        self.pixels = cp.empty(MODEL_INPUT_SHAPE, dtype=cp.float32)
        self.embedding = cp.empty(EMBEDDING_SHAPE, dtype=cp.float32)
        self.map_ready = cp.cuda.Event(disable_timing=True)
        self.preprocess_start = cp.cuda.Event()
        self.preprocess_done = cp.cuda.Event()
        self.embedding_done = cp.cuda.Event()

        ort.preload_dlls(directory="")
        device = cp.cuda.Device().id
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            str(model), sess_options=options,
            providers=[("CUDAExecutionProvider", {
                "device_id": device, "user_compute_stream": str(self.stream.ptr), "use_tf32": 0,
            })],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("The encoder requires the CUDA execution provider")
        self.binding = self.session.io_binding()
        self.binding.bind_input(
            "pixel_values", "cuda", device, np.float32, self.pixels.shape, self.pixels.data.ptr,
        )
        self.binding.bind_output(
            "embedding", "cuda", device, np.float32, self.embedding.shape, self.embedding.data.ptr,
        )
        self.run_options = ort.RunOptions()
        self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

        self.records: list[dict] = []
        self.embeddings: list[np.ndarray] = []
        self._pending: dict | None = None
        self._window_id = -1
        self._next_sequence = 0
        self._complete = False
        self._first_scheduled_ns = 0

    def reference_embedding(self, power: np.ndarray) -> np.ndarray:
        """Encode a serialized reference map outside the scheduled trial."""
        if self._pending is not None:
            raise RuntimeError("The inference handoff is already in use")
        with self.stream:
            self.handoff.set(np.ascontiguousarray(power), stream=self.stream)
            self._enqueue_encoder()
        self.embedding_done.synchronize()
        return self.embedding.get(stream=self.stream)

    def add_batch(self, power: cp.ndarray, jobs: list[dict], consumed_ns: int, stream: cp.cuda.Stream):
        """Accumulate ordered batch results on the caller's imaging stream."""
        first = jobs[0]
        window_id = first["sequence"] // self.window_frames
        if window_id != self._window_id:
            self._window_id = window_id
            self._next_sequence = window_id * self.window_frames
            self._complete = True
            self._first_scheduled_ns = first["scheduled_ns"]
            self.accumulator.fill(0)

        for job in jobs:
            sequence = job["sequence"]
            if sequence != self._next_sequence:
                self._complete = False
            self._next_sequence = sequence + 1
        if not self._complete:
            return
        cp.add(self.accumulator, power.reshape(self.accumulator.shape), out=self.accumulator)

        if self._next_sequence % self.window_frames == 0:
            # One handoff slot bounds the pipeline. Keep the completed
            # accumulator intact until any previous embedding has retired.
            self.drain()
            cp.copyto(self.handoff, self.accumulator)
            self.map_ready.record(stream)
            self._pending = {
                "window_id": window_id,
                "first_sequence": window_id * self.window_frames,
                "last_sequence": self._next_sequence - 1,
                "first_scheduled_ns": self._first_scheduled_ns,
                "last_scheduled_ns": jobs[-1]["scheduled_ns"],
                "last_consumed_ns": consumed_ns,
            }

    def _enqueue_encoder(self):
        self.preprocess_start.record(self.stream)
        preprocess_power_map(self.handoff, self.scale, out=self.pixels)
        self.preprocess_done.record(self.stream)
        self.session.run_with_iobinding(self.binding, self.run_options)
        self.embedding_done.record(self.stream)

    def poll(self) -> bool:
        """Advance the handoff or retire a completed embedding without waiting."""
        pending = self._pending
        if pending is None:
            return False
        if "inference_submit_ns" not in pending:
            if not self.map_ready.done:
                return False
            window_id = pending["window_id"]
            pending["window_ready_ns"] = time.perf_counter_ns()
            if self.trace:
                cp.cuda.nvtx.Mark(f"exercise7/window_ready/{window_id}")
            with self.stream:
                self.stream.wait_event(self.map_ready)
                pending["inference_submit_ns"] = time.perf_counter_ns()
                if self.trace:
                    cp.cuda.nvtx.RangePush(f"exercise7/inference_submit/{window_id}")
                try:
                    self._enqueue_encoder()
                finally:
                    if self.trace:
                        cp.cuda.nvtx.RangePop()
            return True
        if not self.embedding_done.done:
            return False

        completed_ns = time.perf_counter_ns()
        pending["embedding_completed_ns"] = completed_ns
        pending["deadline_miss"] = completed_ns - pending["last_scheduled_ns"] > self.deadline_ns
        pending["gpu_preprocess_ms"] = float(cp.cuda.get_elapsed_time(self.preprocess_start, self.preprocess_done))
        pending["gpu_inference_span_ms"] = float(cp.cuda.get_elapsed_time(self.preprocess_done, self.embedding_done))
        self.embeddings.append(self.embedding.get(stream=self.stream))
        self.records.append(pending)
        self._pending = None
        if self.trace:
            cp.cuda.nvtx.Mark(f"exercise7/embedding_delivered/{pending['window_id']}")
        return True

    def drain(self) -> None:
        """Wait when the single handoff must be reused or the replay is ending."""
        if self._pending is not None:
            if "inference_submit_ns" not in self._pending:
                self.map_ready.synchronize()
                self.poll()
            self.embedding_done.synchronize()
            self.poll()
