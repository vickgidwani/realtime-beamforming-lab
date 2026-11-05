# Exercises

The labs are designed to make each performance claim traceable to a baseline, workload, and profiler signal. Record actual hardware and software metadata with every run. A lower time is not assumed in advance; some exercises are successful when they expose a bottleneck or a scaling limit.

## Available now

1. [Baseline and transfer path](01-transfer-path.md) — compare host-array and GPU-resident processing on the PyMUST phantom.
2. [Reduce before copying to host](02-device-side-reduction.md) — quantify the output-transfer reduction from GPU-side Power Doppler accumulation.

## Planned extensions

3. [RF-to-IQ preprocessing scale](03-rf-to-iq-scaling.md) — measure CPU worker and dtype choices while checking numerical error.
4. [Copy/compute overlap and tail latency](04-streaming-tail-latency.md) — replay a sustained stream and measure overlap, queue depth, frame drops, and deadlines.
5. [Custom DAS kernel](05-custom-das-kernel.md) — implement and profile a CUDA reference kernel, then tune memory access and work partitioning against `mach`.

The active dataset and transformations are documented in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md). Each future dataset adapter should add its provenance before adding benchmark results.
