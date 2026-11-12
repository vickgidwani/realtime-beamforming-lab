# Mach async prototype patch

These experimental patches target the Forest-Neurotech Mach `v0.2.1` source
tag (tag commit `237d38158bf299f67b46d71eaba2cdeac9c6f02a`).

`mach-v0.2.1-async-synchronize.patch` adds an opt-in `synchronize=False`
argument to `mach.kernel.beamform()`.

The default remains synchronous. The opt-in path is restricted to CUDA arrays
that already have the expected dtypes and contiguous layouts. It enqueues on
CUDA's default stream, checks immediate launch errors, and returns without a
device-wide wait. Callers must keep all arrays alive and order dependent GPU
work after the beamform on the same stream. CPU or mixed inputs remain
synchronous and are rejected if asynchronous mode is requested.

This is a lab prototype, not an upstream Mach change. To apply it to a clean
checkout of the pinned source tag:

```bash
git checkout --detach 237d38158bf299f67b46d71eaba2cdeac9c6f02a
git apply --unidiff-zero /path/to/realtime-beamforming-lab/patches/mach-v0.2.1-async-synchronize.patch
```

`mach-v0.2.1-explicit-stream-async.patch` is the follow-up patch. It includes
the asynchronous argument and adds an explicit `stream` argument to the Python
wrapper. The CUDA kernel launches on that stream; synchronous calls wait for
that stream, while asynchronous calls return after launch-error checking. For
explicit-stream calls, arrays are exchanged through DLPack with `stream=-1`,
which deliberately disables automatic stream handoff. The caller must ensure
all input writes are complete on the supplied stream and order later consumers
after the beamform. The lab runner queues the H2D completion event on the shared
compute stream before Mach, then records its own completion event after the
power reduction. Both frame slots share one compute stream, so this experiment
tests copy/compute overlap without running beamform kernels concurrently.

Apply the follow-up patch to a clean checkout with:

```bash
git checkout --detach 237d38158bf299f67b46d71eaba2cdeac9c6f02a
git apply /path/to/realtime-beamforming-lab/patches/mach-v0.2.1-explicit-stream-async.patch
```

Build and numerical/stream-ordering checks are recorded in Exercise 5 after
the prototype has been run on the Colab T4.
