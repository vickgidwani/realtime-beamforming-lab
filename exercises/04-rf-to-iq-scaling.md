# Exercise 4: RF analytic and baseband conversion

**Status: notebook prepared; Colab T4 run pending.** The notebook compares two ways to form complex RF data, CPU and GPU implementations, and scaling across measured and synthetic workloads.

## Questions

- How do the Hilbert analytic-signal path and the baseband mix-and-filter path compare in speed, memory use, and image accuracy?
- When does GPU conversion or CPU worker parallelism help?
- How do those results change between a short slow-time ensemble and a larger multi-angle workload?

## Workloads

- **PyMUST rotating-disk phantom:** measured raw RF with 128 elements, 334 fast-time samples, and 32 slow-time frames. Its sampling and center frequencies are available, and `pymust.rf2iq` provides the CPU baseband-IQ reference.
- **PICMUS resolution/distortion phantom:** measured raw RF with 128 elements, 3,328 samples, 75 transmit angles, and one compounded imaging frame. The angles are independent transmit acquisitions, not slow-time frames. Mach's UFF adapter applies a Hilbert transform when the UFF modulation-frequency field is zero. For the baseband path, obtain the actual carrier frequency from pulse/acquisition metadata; do not treat that zero as the carrier.
- **Larger scale case:** use deterministic synthetic RF with known envelope and phase, or a clearly labeled replay of measured traces. Keep real-data accuracy checks separate from scale-only runs.

## Comparison

Run the same raw RF array through each method on each dataset:

| Method | CPU implementation | GPU implementation |
| --- | --- | --- |
| Analytic RF | `scipy.signal.hilbert` | `cupyx.scipy.signal.hilbert` |
| Baseband IQ | `pymust.rf2iq` on PyMUST data; matched SciPy mix/filter on PICMUS; trace-chunked worker sweep | CuPy mix and matching low-pass/filter steps |

Use a vectorized, single-worker CPU baseline, then sweep a small number of CPU workers by chunking independent traces across receive elements, frames, or transmit angles. Keep each complete fast-time trace in one chunk; do not split a trace across workers. Record dtype and use matched output precision for CPU/GPU comparisons.

Load each dataset once and time file I/O separately from conversion. Report conversion compute time, required host/device transfers, beamforming time, and conversion-to-image wall time separately. The notebook samples peak host RSS and CuPy-pool use as approximate deltas and records raw/output payload bytes exactly. Include transfer costs in end-to-end comparisons; do not count the UFF read or geometry setup as RF conversion.

Hilbert produces analytic RF with the carrier still present, while mix-and-filter produces baseband IQ. For cross-method accuracy, account for that carrier rotation and match bandwidth and amplitude conventions. Also compare beamformed outputs using the corresponding Mach modulation-frequency setting: zero for undemodulated analytic RF, and the actual center frequency for baseband IQ.

## Record

- Conversion latency and throughput, CPU worker count, and dtype.
- Sampled peak host RSS and GPU pool memory deltas, raw/output payload bytes, and transfer bytes.
- Normalized complex-signal error for CPU versus GPU within each method; phase-aligned error between methods; and final beamformed-image error.
- Measured versus synthetic/replay input, plus sample, element, frame, and transmit dimensions.

The small PyMUST case may show overhead rather than speedup. A useful outcome is identifying the worker-count and workload-size crossover, whether either method reduces memory or end-to-end time, and whether the numerical differences affect the final image.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom) and [PICMUS provenance](../DATA_PROVENANCE.md#active-dataset-picmus-experimental-resolutiondistortion-phantom).
