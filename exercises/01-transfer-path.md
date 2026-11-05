# Exercise 1: transfer path

## Question

How much of this workload is spent moving input and geometry to the GPU, and what changes when IQ and static geometry stay GPU-resident?

## Baseline and comparison

Run the same dataset, grid, batch size, and number of measured runs in both modes:

```bash
beamform-lab benchmark --mode numpy --runs 200 --warmup 10 --batch-size 32 --output artifacts/host-arrays.json
beamform-lab benchmark --mode cupy --runs 200 --warmup 10 --batch-size 32 --output artifacts/gpu-resident.json
```

Compare correctness first (`mean_output_feature` and the saved maps), then pipeline wall time, upload time, CUDA-event time, and p50/p95/p99 latency. Profile a short run with Nsight Systems and inspect H2D/D2H copies, repeated geometry transfers, synchronization, and gaps between kernels.

## Success criteria

- Both modes produce numerically comparable Power Doppler output.
- The trace establishes which arrays move, how often, and how much data is transferred.
- The report records the dataset fingerprint and GPU/software environment.

A lower wall time is a hypothesis, not a guaranteed result; the sample is small and transfer/runtime overhead varies by GPU and driver.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
