# Exercise 7 — GPU-Resident Inference from Phantom Replay

The first 1 kHz integrated replay is runnable. It reuses the [scheduled phantom replay](05-streaming-tail-latency.md) and the ONNX Runtime I/O Binding pattern from the GPU Inference Serving lab.

## Exercise 5 handoff

Use the [fastest measured L4 setup](05-streaming-tail-latency.md#fastest-measured-setup) as the imaging baseline. Keep the incoming IQ in host RAM, stage it asynchronously, and retain the beamformed IQ and power on the GPU through inference. The 10 kHz Exercise 5 run peaked at 9,646 completed frames/s with a 100% drop-or-late share. Start the integrated parity run at a lower offered rate before testing overload.

Exercise 5 accumulated all completed batches into one map. Exercise 7 needs a separate map per window, with explicit sequence bounds and completeness. Use NVTX to mark `window_ready` after a complete power map is available and `inference_submit` at the ONNX Runtime handoff. The GPU timeline can then distinguish a wait for a complete window from a delay after the window is ready.

## Interview prompt

A decoder meets its latency budget on an idle GPU. What changes when beamforming, imaging preprocessing, and inference share memory and compute on that GPU?

## First run

1. Replay the PyMUST phantom with its existing scheduled-arrival runner. Use one fixed-length ensemble and one output power map per complete window. Reuse the existing RF-to-IQ preparation for the first pass and label that timing as scheduled IQ replay to output. Add RF-to-IQ inside the timed path as a follow-up.
2. Keep beamformed IQ, the power map, and the inference input on CUDA. Start with the existing power reduction. Add a separately timed, fixed-window clutter filter once the device handoff works. Specify the window, shape, dtype, layout, device, timestamps, and buffer ownership at the boundary. Order the consumer with CUDA events and release each slot only after the consumer completes. Inspect the trace for host round trips and unexpected synchronization.
3. Run the same windows through a serialized reference and the integrated path. Benchmark the decoder alone with matching device inputs, then compare it with inference during concurrent imaging. Fix the engine, device, and workload per comparison. Sweep scheduled arrival rate at a fixed queue bound and declare the deadline and overload policy before running.

## Window and drop policy for the first run

Start at **1,000 offered frames/s** with nonoverlapping 32-frame windows aligned to the offered sequence numbers. Window 0 covers frames 0–31, window 1 covers 32–63, and so on. Each window spans 31 ms from its first scheduled arrival to its last, and a new window begins every 32 ms. The clean parity run should complete every window and produce one embedding per window, with zero dropped frames or windows. The deadline is **100 ms after the last frame's scheduled arrival**.

If any frame drops, invalidate its entire window and skip inference for that power image. Keep the next window on its original sequence boundary. For example, if frame 17 drops, window 0 produces no image or embedding and window 1 still starts at frame 32. Once the drop is known, skip further work for that window where possible. Let already-submitted GPU work finish before reusing its buffers.

Count input-frame drops against offered frames and invalid windows against offered windows. Record the invalid window's sequence range and reason in the run report. Test this deliberately after the clean run passes.

### Inference handoff and timing

Use one reusable GPU handoff slot and at most one inference in flight. The slot holds the CUDA arrays used to turn a completed power map into a model input and to receive its embedding. Imaging signals when the window is ready. Inference waits for that event, and the slot is released after inference completion and any downstream reads from the slot. The beamforming staging buffers and the next window's power accumulator have separate lifetimes. If the handoff slot is occupied, the next completed map must wait within the pipeline's buffer bound and may not overwrite the slot.

Deliver every produced embedding from a complete window and tag it with `deadline_miss` if its completion timestamp exceeds the deadline. Record the first and last scheduled frame timestamps, when the last frame is consumed by the replay dispatcher, when the complete power map becomes ready, inference submission, and when the host observes the CUDA completion event for the embedding. Use the monotonic host clock for end-to-end intervals. Report p50, p95, p99, and max for first-frame age, last-scheduled-frame-to-embedding latency, and last-consumed-frame-to-embedding latency. Use the scheduled timestamp for deadline scoring so time spent waiting upstream remains visible. Keep missing-window counts in the offered-window denominator.

## Stand-in encoder

Use the pinned [Microsoft ResNet-50](https://huggingface.co/microsoft/resnet-50) from the GPU Inference Serving lab. In that checkout, with its export dependencies installed, run `inference-lab-export --embedding-output artifacts/resnet50-embedding.onnx`. Keep the ONNX file and any accompanying external weight file together.

The encoder accepts contiguous FP32 `pixel_values` of shape `[1, 3, 224, 224]` and returns one FP32 `embedding` of shape `[1, 2048]`. The vector is the globally pooled feature map before the classifier. It provides a repeatable pretrained inference workload for orchestration and timing.

The [CUDA adapter](../src/beamforming_lab/beamformed_image_preprocessing.py) maps nonnegative power `p` to `p / (p + s)`, where `s` is a fixed positive scale. A value equal to `s` maps to 0.5, and large values approach 1 smoothly. Use the 99th percentile of the 32-frame reference power map as the calibration scale and keep it fixed during replay. The L4 PyMUST smoke check measured `s = 4,342,520,320` on the 251 × 251 grid. Apply the same rule in the serialized reference.

The adapter runs on the caller's CUDA stream and writes into the handoff slot's preallocated FP32 model input. It uses center-aligned bilinear resize to 224 × 224, replicates the grayscale value across three channels, and applies the pinned ResNet channel normalization. The inference-serving CUDA extension accepts decoded RGB bytes. This adapter uses the same normalization constants directly on the float32 power values and avoids an 8-bit intermediate.

The scheduled runner binds the model input and output on CUDA with ONNX Runtime. It accumulates each complete window on the imaging stream, then hands the GPU-resident map to the inference stream. It saves one 2,048-value embedding per valid window keyed by window ID. The final embedding is copied to the host for the lab output file after inference completes. IQ, the power map, and model input stay on the GPU after the batch H2D copy.

## Run the first L4 replay

```bash
PYTHONPATH="$(cat artifacts/mach-prepared-path.txt)" \
    .venv/bin/python scripts/replay_windowed_inference.py \
    --rate-hz 1000 --frames 1024 --deadline-ms 100 \
    --output artifacts/exercise7/replay-1khz.json
```

On a replacement VM, use the prepared Mach build under `artifacts/mach-async` provisioned with `--patched-mach`. The JSON records timestamps, deadline flags, dropped frames, invalid windows, parity, and latency distributions. The adjacent `.npz` contains the delivered embeddings and their window IDs. Add `--trace-stages` for an Nsight Systems capture or `--omit-sequence 17` for a separate missing-frame run.

### First L4 result

The unprofiled 1 kHz run processed 1,024 offered IQ frames and delivered all 32 embeddings with zero drops or deadline misses. Maximum relative embedding error against the serialized reference was 2.82 × 10⁻⁵ with TF32 requested off.

| Latency measured on the host | p50 | p95 | Max |
| --- | ---: | ---: | ---: |
| First scheduled frame to embedding | 36.58 ms | 36.73 ms | 37.20 ms |
| Last scheduled frame to embedding | 5.58 ms | 5.73 ms | 6.20 ms |
| Last consumed frame to embedding | 4.19 ms | 4.33 ms | 4.74 ms |

The eight-window Nsight capture showed host readback of only the final 8 KiB embedding per window. Omitting frame 17 or 31 in separate four-window runs invalidated window 0 and preserved window 1's frame-32 boundary. With a deliberately tight 0.1 ms test deadline, all four complete embeddings were still delivered and flagged late. A separate two-map test verified that reusing the one handoff slot preserved the association between each power map and its embedding.

## Follow-up: time RF arrival through inference

The PyMUST file contains raw RF, and the current loader converts it to baseband IQ on the CPU with `pymust.rf2iq` before the replay clock starts. A scanner may expose raw RF or already-demodulated IQ at its software boundary. For the raw-RF case, rerun the same scheduled workload with RF-to-IQ inside the timed pipeline. Compare conversion time, frame and window drops, embedding age, and remaining margin under the 100 ms deadline against the prepared-IQ baseline.

## Success criteria

- Match the serialized reference on complete windows within a declared numerical tolerance. State how gaps invalidate a window and show that a dropped frame cannot produce a misleadingly fresh output.
- Report first-input-to-output age, including window fill, and last-input-to-output latency. Include p50, p95, p99, max, and stage timings. Count drops, incomplete windows, and late outputs against the offered workload and define every denominator.
- Save an unprofiled latency report and a short Nsight Systems trace. Explain any change in inference tail latency using the shared GPU timeline. Document the highest tested load that meets the chosen budget and show what happens above it.

## Fault-injection drill

Run this after the clean replay passes output parity at a fixed offered rate and queue bound. Use the same replay and schedule for all trials, and inject one fault per run.

1. **Missing frame.** Omit one known sequence number while preserving its scheduled slot. Derive the affected windows from their sequence bounds and decide what the consumer receives for each invalid window. Continue the replay until the first complete, fresh output.
2. **GPU stall.** Delay completion of a selected GPU work item beyond its deadline using a bounded test hook or controlled GPU contention in the replay harness. Verify the delay in the trace. Exercise the queue bound, stale-output policy, and recovery after the stall clears.

Before either run, check that the output carries its sequence range, completion timestamp, and deadline flag. Assert that a missing frame cannot silently shift subsequent window alignment, late embeddings from complete windows are delivered with the deadline flag, and a buffer stays owned until the GPU consumer completes.

Save the baseline and fault reports with fault sequence, affected window IDs, drops, deadline misses, peak queue depth, and time to the first fresh, valid output. Explain what a downstream consumer sees during the fault and recovery.
