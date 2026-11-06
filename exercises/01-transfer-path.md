# Exercise 1: transfer path

## Question

How much of this workload is spent moving input and geometry to the GPU, and what changes when IQ and static geometry stay GPU-resident?

## Hypothesis

Both modes call mach's CUDA beamformer, so this is a comparison of data placement and transfer paths rather than CPU versus GPU compute. RF-to-IQ conversion happens on the CPU in both modes. I expect the CuPy path to reduce repeated host/device transfer overhead by keeping static geometry GPU-resident and uploading each new IQ ensemble once per call. Geometry is uploaded once per prepared CuPy geometry and reused while the probe and scan configuration stay fixed. I expect numerically comparable Power Doppler maps. The workload is small, so beamforming and reduction may dominate and a speedup is not guaranteed.

In NumPy mode, mach stages host IQ and geometry to temporary GPU buffers on each call, then returns NumPy output to the host; the Power Doppler reduction runs on the CPU. In CuPy mode, mach receives GPU-resident arrays, the reduction stays on the GPU, and the reduced map plus a scalar summary return to the host. CuPy does not retain IQ across calls: the IQ array is uploaded again for each ensemble.

Here, Power Doppler reduction means taking the squared magnitude of each beamformed complex-IQ frame and summing across the 32-frame ensemble, leaving one power value per grid point. The end-to-end timing includes this reduction, but does not time it separately.

## Baseline and comparison

Run the same dataset, grid, batch size, and number of measured runs in both modes. The notebook uses 200 measured runs, 10 warmups, a 251 × 251 grid, and batch size 32, then checks the saved maps and plots the latency distributions:

```bash
beamform-lab benchmark --dataset pymust --mode numpy --measure-cold-start --runs 200 --warmup 10 --batch-size 32 --grid-size 251 --output results/exercise1-pymust-numpy.json --save-power artifacts/exercise1-pymust-numpy-power.npy
beamform-lab benchmark --dataset pymust --mode cupy --measure-cold-start --runs 200 --warmup 10 --batch-size 32 --grid-size 251 --output results/exercise1-pymust-cupy.json --save-power artifacts/exercise1-pymust-cupy-power.npy
```

`--measure-cold-start` records device setup plus the first Power Doppler result before warmups. NumPy's first CUDA use occurs inside its first mach call; CuPy's can occur during geometry preparation. Dataset loading and RF-to-IQ conversion happen before this timer. The notebook measures a shared CPU RF-to-IQ p50 separately and does not add it to the cold or warmed path timings. For a robust cold-start comparison, repeat each mode in several fresh processes and alternate the order.

Compare correctness first (`mean_output_feature` and the saved maps), then cold setup plus first result and warmed ensemble pipeline wall time. Both pipeline timers include data movement, beamforming, Power Doppler reduction, and output handling. Since NumPy reduces on the CPU and CuPy on the GPU, and reduction is not timed separately, the end-to-end speedup cannot be attributed to beamforming or reduction alone. The report's explicit input-upload timer applies to CuPy; NumPy's host-array transfer behavior is inside mach's call. Profile a short run with Nsight Systems to inspect H2D/D2H copies, repeated geometry transfers, synchronization, and gaps between kernels.

## Success criteria

- Both modes produce numerically comparable Power Doppler output.
- The trace establishes which arrays move, how often, and how much data is transferred.
- The report records the dataset fingerprint and GPU/software environment.

A lower wall time is a hypothesis, not a guaranteed result; the sample is small and transfer/runtime overhead varies by GPU and driver.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
