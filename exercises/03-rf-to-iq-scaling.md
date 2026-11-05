# Exercise 3: RF-to-IQ preprocessing scale

**Status: planned.** The current CLI measures RF-to-IQ conversion once during workload setup; it does not yet sweep preprocessing workers or implementations.

## Question

Is RF-to-IQ conversion a meaningful part of end-to-end latency, and does parallelizing independent frame chunks help at realistic ensemble sizes?

## Proposed experiment

Add a separate preprocessing benchmark that compares one worker with a small worker-count sweep and, if implemented, CPU versus GPU demodulation. Test the current 32-frame sample and a larger, clearly labeled replay workload. Keep acquisition geometry and frame order fixed. Validate each result against the reference IQ array and final beamformed output before comparing performance.

## Record

- preprocessing time separately from beamforming;
- worker count, dtype, allocation count/bytes, CPU utilization, and total wall time;
- normalized IQ error and final image error versus the reference;
- whether the workload is measured data or a repeated synthetic replay.

The small sample may show overhead rather than a speedup. A useful outcome is identifying the worker-count crossover or showing that preprocessing is not the critical path.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
