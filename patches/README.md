# Mach async prototype patch

`mach-v0.2.1-async-synchronize.patch` is an experimental patch against the
Forest-Neurotech Mach `v0.2.1` source tag (tag commit
`237d38158bf299f67b46d71eaba2cdeac9c6f02a`). It adds an opt-in
`synchronize=False` argument to `mach.kernel.beamform()`.

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

Build and numerical/stream-ordering checks are recorded in Exercise 5 after
the prototype has been run on the Colab T4.
