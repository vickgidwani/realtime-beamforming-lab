# Exercise 2: reduce before copying to host

## Question

Can the pipeline reduce beamformed frames to a Power Doppler map on the GPU and copy back only the final map?

## Baseline and comparison

The current `numpy` and `cupy` paths place the ensemble reduction on different sides of the device boundary. Run both with identical inputs and profile the output path. For the default shape, the full complex64 beamformed result is approximately 16.1 MB (`63,001 × 32 × 8` bytes); a float32 Power Doppler map is approximately 0.252 MB (`63,001 × 4` bytes). The output-transfer payload can therefore be 64 times smaller when the reduction happens before D2H transfer.

Inspect transfer sizes and locations in Nsight Systems. Compare output maps numerically and record the measured D2H bytes and elapsed time.

## Success criteria

- The report documents the byte-count calculation and confirms the observed transfer path in the trace.
- The GPU-reduced output matches the reference within a stated tolerance.
- The benchmark JSON and trace are associated with the same commit and environment.

The byte reduction follows from the array shapes; the resulting speedup depends on device and transfer overhead.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
