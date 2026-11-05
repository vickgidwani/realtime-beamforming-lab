# Exercise 4: streaming overlap and tail latency

**Status: planned.** The current benchmark repeats complete ensembles; it does not model a live producer/consumer queue.

## Question

Can the pipeline sustain the acquisition rate while overlapping host-to-device copies and beamforming, and what happens to tail latency and queue depth near saturation?

## Proposed experiment

Add a bounded replay source and process frames in chunks using pinned host buffers, asynchronous copies, and double buffering. Compare synchronous copies with copy/compute overlap while keeping the same beamforming parameters. Start with recorded phantom frames, then increase load using an explicitly labeled replay multiplier or a larger channel dataset.

Record per-frame arrival, queue, start, and completion timestamps. Sweep input rate and batch size. Report p50/p95/p99/max end-to-end latency, deadline misses, queue depth, and dropped frames. Use Nsight Systems to verify that copy/compute overlap actually appears in the CUDA timeline.

## Success criteria

- The input rate and deadline are stated explicitly.
- Queue growth and frame-drop behavior are bounded and reproducible.
- Profiling overhead is excluded from the final latency numbers.
- Replayed/synthetic load is distinguished from a real scanner acquisition.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
