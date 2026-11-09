# Exercise 6: custom delay-and-sum kernel

**Status: planned.** The current project calls the prebuilt `mach` kernel; its internal block and memory-access implementation is not tuned by this repository.

## Question

Can a small, understandable CUDA delay-and-sum implementation match the reference output, and which thread mapping and memory layout provide the best performance on a fixed GPU?

## Proposed experiment

1. Implement a simple correctness-first DAS kernel for a deliberately small scan grid.
2. Compare its complex beamformed output against `mach` using absolute/relative error and image-level metrics.
3. Scale points, elements, and frames, then profile a few block sizes and thread mappings with Nsight Compute.
4. Use Nsight Systems to include launch, allocation, and transfer overhead in the pipeline view; use Nsight Compute to inspect kernel-level memory throughput and occupancy.

## Optional extension: image quality versus compute

Once the reference kernel can expose the delay-compensated receive-aperture samples for each pixel, compare DAS with three methods represented in the PICMUS results. Keep this as a separate experiment from kernel tuning: report each method's image-quality metrics and full processing time, rather than attributing an algorithmic change to a faster DAS kernel.

1. **Phase-coherence weighting (smallest step).** Compute phase coherence (PCF) or sign coherence (SCF) across the delayed receive channels and use it to weight the DAS output. Start here because it adds a per-pixel statistic and weight without replacing the DAS reconstruction.
2. **Basic sparse regularization (introductory approximation).** Apply an explicit L1 prior in a wavelet domain to a small DAS envelope image, using proximal soft-thresholding. Treat this as image-domain regularization inspired by PICMUS sparse methods, not a reproduction of the Besson submission or a channel-domain inverse solver. Inspect how the threshold changes point-target resolution, background speckle, and contrast.
3. **Minimum-variance / MVDR (advanced).** Estimate a receive covariance matrix from overlapping subarrays of delayed channel samples, add diagonal loading for stability, solve for distortionless adaptive weights, and apply them to the samples. Begin with a small region of interest and a reduced aperture; compare the cost and output as subarray length and loading change.

Use the PICMUS resolution phantom first for a bounded comparison, then add a contrast phantom before drawing broader conclusions. Where practical, compare 1, 11, and 75 transmit angles. Keep the selected dataset, angle set, scan grid, and output scaling identical across methods. Record at least axial/lateral FWHM on point targets, a contrast metric on a cyst phantom, and a speckle-preservation measure; report the chosen regions of interest and parameter settings. Separate delay/sample preparation, reconstruction or regularization, envelope formation, transfers, and total wall time. The official challenge ranked image quality, not runtime, so our timing results will be hardware-specific additions.

The official result gallery labels Cohen and Shoham's entry as phase-coherence based and includes sparse-regularization and minimum-variance entries. For background, see [Camacho et al., *Phase Coherence Imaging*](https://pubmed.ncbi.nlm.nih.gov/19473914/), [Besson et al., *Morphological component analysis for sparse regularization in plane wave imaging*](https://infoscience.epfl.ch/record/221270/files/PICMUS_IUS2016.pdf), and [Rindal and Austeng, *Minimum variance beamformers for coherent plane-wave compounding*](https://api.repository.cam.ac.uk/server/api/core/bitstreams/16d7bed7-7f39-4a70-a6dd-2c8d41260d00/content). The sparse and MV methods above are deliberately simplified learning exercises, not faithful reproductions of those submissions.

## Success criteria

- The reference output and numerical tolerance are documented before optimization.
- Every measured version has identical inputs and output shape.
- Kernel-only time is reported separately from setup, copies, and Power Doppler reduction.
- Any speedup is reported only for the named GPU and workload, with the trace and runtime provenance retained.
- For the optional image-quality extension, image quality and runtime are reported separately, and each simplified method is identified as such.

The dataset source and transformation history are in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom). The current Mach kernel remains the correctness and performance reference.
