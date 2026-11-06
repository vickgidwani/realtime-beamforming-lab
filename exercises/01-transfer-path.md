# Exercise 1: transfer path

## Question

How much of this workload is spent moving input and geometry to the GPU, and what changes when IQ and static geometry stay GPU-resident?

## Hypothesis

Both modes call mach's CUDA beamformer, so this is a comparison of data placement and transfer paths rather than CPU versus GPU compute. RF-to-IQ conversion happens on the CPU in both modes. I expect the CuPy path to reduce repeated host/device transfer overhead by keeping static geometry GPU-resident and uploading each new IQ ensemble once per call. Geometry is uploaded once per prepared CuPy geometry and reused while the probe and scan configuration stay fixed. I expect numerically comparable Power Doppler maps. The workload is small, so beamforming and reduction may dominate and a speedup is not guaranteed.

In NumPy mode, mach stages host IQ and geometry to temporary GPU buffers on each call, then returns NumPy output to the host; the Power Doppler reduction runs on the CPU. In CuPy mode, mach receives GPU-resident arrays, the reduction stays on the GPU, and only the final map is copied back. CuPy does not retain IQ across calls: the IQ array is uploaded again for each ensemble.

## Baseline and comparison

Run the same dataset, grid, batch size, and number of measured runs in both modes. The notebook uses 200 measured runs, 10 warmups, a 251 × 251 grid, and batch size 32, then checks the saved maps and plots the latency distributions:

```bash
beamform-lab benchmark --dataset pymust --mode numpy --measure-cold-start --runs 200 --warmup 10 --batch-size 32 --grid-size 251 --output results/exercise1-pymust-numpy.json --save-power artifacts/exercise1-pymust-numpy-power.npy
beamform-lab benchmark --dataset pymust --mode cupy --measure-cold-start --runs 200 --warmup 10 --batch-size 32 --grid-size 251 --output results/exercise1-pymust-cupy.json --save-power artifacts/exercise1-pymust-cupy-power.npy
```

`--measure-cold-start` records the first device setup plus first Power Doppler result before warmups. NumPy's first CUDA use occurs inside its first mach call; CuPy's can occur during geometry preparation. The report keeps that one cold observation separate from the warmed per-ensemble percentiles. Dataset loading and RF-to-IQ conversion happen before this timer; the reported cold first-sample estimate adds those measured/estimated stages back serially. For a robust cold-start comparison, repeat each mode in several fresh processes and alternate the order.

Compare correctness first (`mean_output_feature` and the saved maps), then cold first-sample cost and warmed ensemble pipeline wall time. The report's explicit input-upload timer applies to CuPy; NumPy's host-array transfer behavior is inside mach's call. Profile a short run with Nsight Systems to inspect H2D/D2H copies, repeated geometry transfers, synchronization, and gaps between kernels.

## Success criteria

- Both modes produce numerically comparable Power Doppler output.
- The trace establishes which arrays move, how often, and how much data is transferred.
- The report records the dataset fingerprint and GPU/software environment.

A lower wall time is a hypothesis, not a guaranteed result; the sample is small and transfer/runtime overhead varies by GPU and driver.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
