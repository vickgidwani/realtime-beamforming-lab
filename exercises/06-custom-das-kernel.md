# Exercise 6: custom delay-and-sum kernel

**Status: planned.** The current project calls the prebuilt `mach` kernel; its internal block and memory-access implementation is not tuned by this repository.

## Question

Can a small, understandable CUDA delay-and-sum implementation match the reference output, and which thread mapping and memory layout provide the best performance on a fixed GPU?

## Proposed experiment

1. Implement a simple correctness-first DAS kernel for a deliberately small scan grid.
2. Compare its complex beamformed output against `mach` using absolute/relative error and image-level metrics.
3. Scale points, elements, and frames, then profile a few block sizes and thread mappings with Nsight Compute.
4. Use Nsight Systems to include launch, allocation, and transfer overhead in the pipeline view; use Nsight Compute to inspect kernel-level memory throughput and occupancy.

## Success criteria

- The reference output and numerical tolerance are documented before optimization.
- Every measured version has identical inputs and output shape.
- Kernel-only time is reported separately from setup, copies, and Power Doppler reduction.
- Any speedup is reported only for the named GPU and workload, with the trace and runtime provenance retained.

The dataset source and transformation history are in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom). The current Mach kernel remains the correctness and performance reference.
