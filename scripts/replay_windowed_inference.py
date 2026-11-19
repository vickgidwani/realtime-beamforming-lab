"""Run scheduled IQ replay through per-window beamforming and CUDA inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from beamforming_lab.beamformer import run_ensemble
from beamforming_lab.streaming_beamformer import StreamingBeamformer
from beamforming_lab.windowed_inference import EMBEDDING_SHAPE, WindowedInference

WINDOW_FRAMES = 32
PARITY_TOLERANCE = 1e-4


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        **{f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)},
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/exercise7/resnet50-embedding.onnx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/exercise7/replay.json"))
    parser.add_argument("--rate-hz", type=int, default=1_000)
    parser.add_argument("--frames", type=int, default=1_024)
    parser.add_argument("--deadline-ms", type=float, default=100)
    parser.add_argument("--omit-sequence", type=int, default=-1)
    parser.add_argument("--trace-stages", action="store_true")
    args = parser.parse_args()
    if args.frames < WINDOW_FRAMES or args.frames % WINDOW_FRAMES:
        parser.error("--frames must be a positive multiple of 32")
    if args.rate_hz <= 0 or not np.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("rate and deadline must be positive and finite")
    if not -1 <= args.omit_sequence < args.frames:
        parser.error("--omit-sequence is outside the replay")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")

    beamformer = StreamingBeamformer()
    if beamformer.frame_count != WINDOW_FRAMES:
        raise RuntimeError("The parity check expects one source ensemble per replay window")
    beamformer.replay_frames = args.frames
    reference = run_ensemble(beamformer.workload, beamformer.geometry, batch_size=1)["power_doppler"]
    scale = float(np.float32(np.percentile(reference, 99)))
    encoder = WindowedInference(
        args.model, reference.shape, scale,
        window_frames=WINDOW_FRAMES, deadline_ms=args.deadline_ms, trace=args.trace_stages,
    )
    expected = encoder.reference_embedding(reference)
    replay = beamformer.run_batched_trial(
        8, args.rate_hz, mode="async_explicit_stream", packer="one_pass",
        concurrent_staging=True, reuse_beamform_output=True, prepare_mach=True,
        trace_stages=args.trace_stages, window_inference=encoder, omit_sequence=args.omit_sequence,
    )

    offered_windows = args.frames // WINDOW_FRAMES
    dropped_by_window = {}
    for drop in replay["drops"]:
        dropped_by_window.setdefault(drop["sequence"] // WINDOW_FRAMES, []).append(drop["sequence"])
    expected_ids = sorted(set(range(offered_windows)) - dropped_by_window.keys())
    actual_ids = [window["window_id"] for window in encoder.records]
    if expected_ids != actual_ids:
        raise RuntimeError(f"Unexpected emitted window IDs: wanted {expected_ids}, received {actual_ids}")

    errors = []
    norm = max(float(np.linalg.norm(expected)), np.finfo(np.float32).tiny)
    for record, embedding in zip(encoder.records, encoder.embeddings, strict=True):
        error = float(np.linalg.norm(embedding - expected) / norm)
        if not np.isfinite(error) or error > PARITY_TOLERANCE:
            raise RuntimeError(f"Window {record['window_id']} failed embedding parity: {error}")
        record["relative_embedding_error"] = error
        errors.append(error)

    def latency(origin: str) -> dict | None:
        return distribution([
            (r["embedding_completed_ns"] - r[origin]) / 1e6 for r in encoder.records
        ])

    report = {
        "gpu": beamformer.cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "offered_rate_hz": args.rate_hz,
        "window_frames": WINDOW_FRAMES,
        "batch_size": 8,
        "queue_capacity_frames": replay["queue_capacity_frames"],
        "deadline_ms_after_last_scheduled_frame": args.deadline_ms,
        "offered_frames": args.frames,
        "dropped_frames": replay["dropped_frames"],
        "offered_windows": offered_windows,
        "produced_embeddings": len(encoder.records),
        "invalid_windows": [
            {
                "window_id": wid,
                "first_sequence": wid * WINDOW_FRAMES,
                "last_sequence": (wid + 1) * WINDOW_FRAMES - 1,
                "missing_sequences": missing,
            }
            for wid, missing in sorted(dropped_by_window.items())
        ],
        "late_embeddings": sum(r["deadline_miss"] for r in encoder.records),
        "power_scale": scale,
        "embedding_relative_tolerance": PARITY_TOLERANCE,
        "max_relative_embedding_error": max(errors, default=None),
        "latency_ms": {
            "first_scheduled_frame_to_embedding": latency("first_scheduled_ns"),
            "last_scheduled_frame_to_embedding": latency("last_scheduled_ns"),
            "last_consumed_frame_to_embedding": latency("last_consumed_ns"),
            "gpu_preprocess": distribution([r["gpu_preprocess_ms"] for r in encoder.records]),
            "gpu_inference_span": distribution([r["gpu_inference_span_ms"] for r in encoder.records]),
        },
        "windows": encoder.records,
        "frame_drops": replay["drops"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    embeddings = (
        np.concatenate(encoder.embeddings, axis=0)
        if encoder.embeddings else np.empty((0, EMBEDDING_SHAPE[1]), dtype=np.float32)
    )
    np.savez(args.output.with_suffix(".npz"), window_ids=np.asarray(actual_ids, dtype=np.int64), embeddings=embeddings)
    print(json.dumps({k: v for k, v in report.items() if k not in {"windows", "frame_drops"}}, indent=2))


if __name__ == "__main__":
    main()
