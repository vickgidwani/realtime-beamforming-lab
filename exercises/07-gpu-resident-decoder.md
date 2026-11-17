# Exercise 7 — GPU-Resident Inference from Phantom Replay

This exercise is planned. Reuse the [scheduled phantom replay](05-streaming-tail-latency.md) and the ONNX Runtime I/O Binding pattern from the GPU Inference Serving lab.

## Interview prompt

A decoder meets its latency budget on an idle GPU. What changes when beamforming, imaging preprocessing, and inference share memory and compute on that GPU?

## First run

1. Replay the PyMUST phantom with its existing scheduled-arrival runner. Use one fixed-length ensemble and a fixed-shape, deterministic stand-in decoder. Reuse the existing RF-to-IQ preparation for the first pass and label that timing as scheduled IQ replay to output. Add RF-to-IQ inside the timed path as a follow-up. The phantom and stand-in model support a systems benchmark.
2. Keep beamformed IQ, the power map, and the inference input on CUDA. Start with the existing power reduction. Add a separately timed, fixed-window clutter filter once the device handoff works. Specify the window, shape, dtype, layout, device, timestamps, and buffer ownership at the boundary. Order the consumer with CUDA events and release each slot only after the consumer completes. Inspect the trace for host round trips and unexpected synchronization.
3. Run the same windows through a serialized reference and the integrated path. Benchmark the decoder alone with matching device inputs, then compare it with inference during concurrent imaging. Fix the engine, device, and workload per comparison. Sweep scheduled arrival rate at a fixed queue bound and declare the deadline and overload policy before running.

## Success criteria

- Match the serialized reference on complete windows within a declared numerical tolerance. State how gaps invalidate a window and show that a dropped frame cannot produce a misleadingly fresh output.
- Report first-input-to-output age, including window fill, and last-input-to-output latency. Include p50, p95, p99, max, and stage timings. Count drops, incomplete windows, and late outputs against the offered workload and define every denominator.
- Save an unprofiled latency report and a short Nsight Systems trace. Explain any change in inference tail latency using the shared GPU timeline. Document the highest tested load that meets the chosen budget and show what happens above it.

## Fault-injection drill

Run this after the clean replay passes output parity at a fixed offered rate and queue bound. Use the same replay and schedule for all trials, and inject one fault per run.

1. **Missing frame.** Omit one known sequence number while preserving its scheduled slot. Derive the affected windows from their sequence bounds and decide what the consumer receives for each invalid window. Continue the replay until the first complete, fresh output.
2. **GPU stall.** Delay completion of a selected GPU work item beyond its deadline using a bounded test hook or controlled GPU contention in the replay harness. Verify the delay in the trace. Exercise the queue bound, stale-output policy, and recovery after the stall clears.

Before either run, define which clock determines freshness, what the output carries for sequence range and validity, and how overlapping windows recover after a gap. Assert that a missing frame cannot silently shift subsequent window alignment, stale output is suppressed or explicitly marked under the chosen policy, and a buffer stays owned until the GPU consumer completes.

Save the baseline and fault reports with fault sequence, affected window IDs, drops, deadline misses, peak queue depth, and time to the first fresh, valid output. Explain what a downstream consumer sees during the fault and recovery.
