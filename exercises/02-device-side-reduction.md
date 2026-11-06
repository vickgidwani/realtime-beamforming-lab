# Exercise 2: reduce before copying to host

The companion [Colab notebook](02-device-side-reduction.ipynb) records the T4 profiling commands, analyzes the saved Nsight Systems exports, checks the output maps, and embeds screenshots from the visualizer.

## Question

Can the pipeline reduce beamformed frames to a Power Doppler map on the GPU and copy back only the final map?

## Baseline and comparison

The current `numpy` and `cupy` paths place the ensemble reduction on different sides of the device boundary. Run both with identical inputs and profile the output path. For the default shape, the full complex64 beamformed result is approximately 16.1 MB (`63,001 × 32 × 8` bytes); a float32 Power Doppler map is approximately 0.252 MB (`63,001 × 4` bytes). The output-transfer payload can therefore be 64 times smaller when the reduction happens before D2H transfer.

Inspect transfer sizes and locations in Nsight Systems. Compare output maps numerically and record the measured D2H bytes and elapsed time.

## Extension: isolate CPU and GPU reduction time

Across the five CuPy trace calls, the median active GPU kernel times for magnitude, power, and summing across frames add up to about 0.89 ms. The 21.4 µs event selected in the screenshot is the separate map transfer. The current benchmark does not time the NumPy reduction by itself.

`benchmark-reduction` generates one beamformed complex64 ensemble on the T4, then uses the same values for both reducers. Its compute-only phase starts with the array already resident in CPU and GPU memory. Its post-beamforming phase compares the practical paths: copy the full ensemble to the CPU and reduce it, or reduce on the GPU and copy the map and scalar summary to the CPU.

Run the unprofiled benchmark for timing distributions:

```bash
beamform-lab benchmark-reduction --runs 20 --warmup 5 --batch-repeats 5 \
  --output results/exercise2-reduction.json
```

For a short Nsight Systems capture with named stage ranges:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=artifacts/exercise2-reduction \
  beamform-lab benchmark-reduction --runs 3 --warmup 2 --batch-repeats 1 --nvtx
```

NVTX ranges are labels around code stages. Nsight Systems displays them on the CPU timeline and correlates GPU kernels launched inside them. Here they label reference setup, CPU/GPU reduction, and each post-beamforming transfer-and-reduce path. The ranges help identify activity in the trace; use the unprofiled JSON for the timing comparison because profiling adds overhead. No Perfetto trace is required.

## Success criteria

- The report documents the byte-count calculation and confirms the observed transfer path in the trace.
- The GPU-reduced output matches the reference within a stated tolerance.
- The benchmark JSON and trace are associated with the same commit and environment.
- The CPU and GPU map reductions use the same beamformed values and agree numerically.
- Compute-only reduction timings are reported separately from each path's required transfer and post-beamforming wall time.

The byte reduction follows from the array shapes; the resulting speedup depends on device and transfer overhead.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
