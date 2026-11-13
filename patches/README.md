# Mach async prototype patch

These experimental patches target the Forest-Neurotech Mach `v0.2.1` source
tag (tag commit `237d38158bf299f67b46d71eaba2cdeac9c6f02a`).

`mach-v0.2.1-explicit-stream-async.patch` adds an opt-in
`synchronize=False` argument and an explicit `stream` argument to
`mach.kernel.beamform()`. It supports both the default-stream asynchronous
path and a caller-selected CUDA stream. Synchronous calls wait for the selected
stream (or the whole device on the legacy default-stream path); asynchronous
calls return after launch-error checking.

The asynchronous and explicit-stream paths require CUDA arrays with the
expected dtypes and contiguous layouts. For explicit-stream calls, the wrapper
passes nanobind adapter objects that implement the DLPack protocol; each adapter
asks CuPy for a capsule using `stream=-1`. Nanobind consumes the protocol object,
while `stream=-1` disables automatic stream handoff. The caller must ensure
input writes are complete on the supplied stream and order later consumers after
beamforming. The lab runner queues the H2D completion event on the shared
compute stream before Mach, then records its own completion event after the
power reduction. Both frame slots share one compute stream, so this experiment
tests copy/compute overlap without running beamform kernels concurrently.

This is a lab prototype, not an upstream Mach change. Apply it to a clean
checkout of the pinned source tag:

```bash
git checkout --detach 237d38158bf299f67b46d71eaba2cdeac9c6f02a
git apply /path/to/realtime-beamforming-lab/patches/mach-v0.2.1-explicit-stream-async.patch
```

Build and numerical/stream-ordering checks are recorded in Exercise 5 after
the prototype has been run on the Colab T4.
