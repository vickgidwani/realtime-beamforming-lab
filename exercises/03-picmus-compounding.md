# Exercise 3: PICMUS multi-angle compounding

## Question

What changes when the beamformer handles 75 transmit angles for one image, and how much of the repeated-run latency comes from host/device movement versus reconstruction?

## Baseline and comparison

Run the same PICMUS UFF file through both array paths:

```bash
beamform-lab benchmark --dataset picmus --mode numpy --runs 20 --warmup 2 \
  --save-image artifacts/picmus-host.npy --output artifacts/picmus-host.json

beamform-lab benchmark --dataset picmus --mode cupy --runs 20 --warmup 2 \
  --save-image artifacts/picmus-device.npy --output artifacts/picmus-device.json
```

Compare the normalized image arrays before comparing latency. Inspect p50/p95/p99 reconstruction latency, the one-time dataset setup and upload costs, and the Nsight Systems trace for per-angle launches, H2D/D2H copies, synchronization, and gaps between work.

```bash
nsys profile --trace=cuda,nvtx,osrt \
  beamform-lab benchmark --dataset picmus --mode cupy --runs 3 --warmup 1
```

## Success criteria

- The report identifies the UFF file, 75-angle input shape, scan grid, source checksum, and runtime environment.
- The saved NumPy and CuPy outputs agree within a stated numeric tolerance after normalization.
- The trace distinguishes one-time input upload from repeated reconstruction and final output transfer.
- The report labels these timings as single-image reconstruction latency.

The file contains one pre-acquired frame, so this exercise does not measure a slow-time fUSI stream or Power Doppler processing. The `numpy` mode still uses mach's CUDA beamformer; it passes host arrays to mach and includes the resulting transfer behavior in repeated-call wall time. The CuPy mode uploads the arrays once, then reuses them. The compounding adapter uses mach's experimental API, which can change between versions.

Dataset details and attribution requirements are documented in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-picmus-experimental-resolutiondistortion-phantom).
