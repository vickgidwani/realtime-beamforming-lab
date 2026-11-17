# Exercises

The labs are designed to make each performance claim traceable to a baseline, workload, and profiler signal. Record actual hardware and software metadata with every run. A lower time is not assumed in advance; some exercises are successful when they expose a bottleneck or a scaling limit.

## Recorded experiments

1. [Baseline and transfer path](01-transfer-path.md) — compare host-array and GPU-resident processing on the PyMUST phantom. The companion [Colab notebook](01-transfer-path.ipynb) runs a matched NumPy/CuPy comparison on a connected T4, including cold-start timing, output checks, and plots.
2. [Reduce before copying to host](02-device-side-reduction.md) — quantify the output-transfer reduction from GPU-side Power Doppler accumulation. The companion [Colab notebook](02-device-side-reduction.ipynb) analyzes the Nsight Systems traces.
3. [PICMUS multi-angle compounding](03-picmus-compounding.md) — profile coherent compounding of 75 experimental phantom transmissions with host and GPU-resident arrays. The companion [Colab notebook](03-picmus-compounding.ipynb) checks image agreement, visualizes timing and data movement, and captures NVTX-labeled Nsight traces.

4. [RF analytic and baseband conversion](04-rf-to-iq-scaling.md) — compare Hilbert and mix-and-filter conversion on CPU and GPU across PyMUST and PICMUS. The [T4 results](../results/exercise4-rf-to-iq.json) include an angle sweep and a synthetic scaling case.
5. [Copy/compute overlap and tail latency](05-streaming-tail-latency.md) — compare serialized and overlapped GPU paths under scheduled phantom replay. The T4 results include queueing, deadlines, drops, and the microbatch follow-up. See also the [Mach stream prototype](../patches/README.md) and [recorded explicit-stream comparison](../results/exercise5-mach-explicit-stream.json).

## Planned exercises

- **Start here for Bridge Realtime.** [Exercise 7 — GPU-resident inference from phantom replay](07-gpu-resident-decoder.md) measures shared-GPU contention and checks recovery from a dropped frame or GPU stall.
- [Exercise 6 — Custom DAS kernel](06-custom-das-kernel.md) implements and profiles a small CUDA reference kernel before testing memory access and work partitioning against Mach. Image-quality variants are optional.

Dataset sources, transformations, checksums, and attribution notes are documented in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md).
