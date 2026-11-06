# Exercises

The labs are designed to make each performance claim traceable to a baseline, workload, and profiler signal. Record actual hardware and software metadata with every run. A lower time is not assumed in advance; some exercises are successful when they expose a bottleneck or a scaling limit.

## Available now

1. [Baseline and transfer path](01-transfer-path.md) — compare host-array and GPU-resident processing on the PyMUST phantom. The companion [Colab notebook](01-transfer-path.ipynb) runs a matched NumPy/CuPy comparison on a connected T4, including cold-start timing, output checks, and plots.
2. [Reduce before copying to host](02-device-side-reduction.md) — quantify the output-transfer reduction from GPU-side Power Doppler accumulation.
3. [PICMUS multi-angle compounding](03-picmus-compounding.md) — profile coherent compounding of 75 experimental phantom transmissions with host and GPU-resident arrays.

## Planned extensions

4. [RF-to-IQ preprocessing scale](04-rf-to-iq-scaling.md) — measure CPU worker and dtype choices while checking numerical error.
5. [Copy/compute overlap and tail latency](05-streaming-tail-latency.md) — replay a sustained stream and measure overlap, queue depth, frame drops, and deadlines.
6. [Custom DAS kernel](06-custom-das-kernel.md) — implement and profile a CUDA reference kernel, then tune memory access and work partitioning against `mach`.

Dataset sources, transformations, checksums, and attribution notes are documented in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md).
