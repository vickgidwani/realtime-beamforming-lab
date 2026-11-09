# Exercise 3: PICMUS multi-angle compounding

## Question

What changes when one image is reconstructed from 75 transmit angles, and how much of repeated-run time comes from host/device movement versus beamforming and compounding?

This is a single-frame, single-image reconstruction exercise. The selected PICMUS UFF file contains 75 steered transmissions that are coherently compounded into one image. It does not measure a slow-time stream, RF-to-IQ conversion, clutter filtering, or Power Doppler reduction.

## Why this uses a different mach API than Exercise 2

Exercise 2 calls [`mach.kernel.beamform`](https://forest-neurotech.github.io/mach/generated/mach.kernel.beamform.html) directly for its single-transmit workload. Exercise 3 uses [`mach.experimental.beamform`](https://forest-neurotech.github.io/mach/generated/mach.experimental.beamform.html), which accepts the transmit dimension and coherently compounds all 75 angles in one Python call. Nsight shows the 75 per-transmit `beamformKernel` launches inside each reconstruction.

That difference is intentional: Exercise 3 asks how input residency affects the multi-angle compounding path. Both modes use the same `experimental.beamform` API; `numpy` stages the four host arrays to CuPy for every reconstruction, while `cupy` uploads them once and reuses the device arrays. This is not a repeat of Exercise 2's direct NumPy-versus-CuPy comparison on the single-transmit kernel.

In mach-beamform 0.2.1, `experimental.beamform` is a Python wrapper around `kernel.beamform`: it loops over the 75 transmits and passes the same output buffer to each call. The CUDA kernel accumulates each transmit's complex contribution into that output with `atomicAdd` on the real and imaginary parts ([wrapper source](https://github.com/Forest-Neurotech/mach/blob/v0.2.1/src/mach/experimental.py#L46-L89), [CUDA accumulation](https://github.com/Forest-Neurotech/mach/blob/v0.2.1/src/mach/kernel.cu#L624-L632)). Thus Exercise 3 applies the same low-level kernel used in Exercise 2 across all transmissions and compounds on the GPU. A hand-written loop would reproduce the wrapper's implementation, not add a distinct benchmark path.

## What differs between the array paths

Both paths use mach's CUDA beamformer and the same PICMUS input. The receive-element and scan coordinates are shared by the transmit angles; angle-specific travel times are stored together in the `tx_wave_arrivals_s` array, with one leading-dimension entry per transmission.

- `numpy` keeps the source arrays on the host between calls and stages all four arrays to CuPy inside each reconstruction. The repeated HtoD copies are part of the measured call. This explicit staging is needed because the installed mach compound API rejects mixed NumPy-input/CuPy-output arguments.
- `cupy` uploads `channel_data`, `rx_coords_m`, `scan_coords_m`, and `tx_wave_arrivals_s` once, then reuses the device arrays for warmup and measured calls.

The report separates dataset loading and setup, the initial CuPy upload, repeated wall and CUDA-event percentiles, and final B-mode output transfer. CUDA-event timing covers work ordered between events; inspect the trace before describing it as kernel-only time. The final-image DtoH time is one measurement per mode; both use the same `cp.asnumpy()` path for the same-size CuPy result, so the recorded difference should be treated as timing variation rather than a NumPy-versus-CuPy output-path advantage.

## Colab run

Open [the companion notebook](03-picmus-compounding.ipynb) from the project checkout on the connected Colab T4. It installs the Colab dependencies, runs both modes (20 measurements after 2 warmups), checks normalized image agreement, plots image and timing results, and captures short NVTX-labeled traces (3 measurements after 1 warmup).

The notebook writes small JSON reports to `results/` and generated images and Nsight reports to the ignored `artifacts/` directory. The first run downloads and verifies the 145.5 MB UFF file; later calls use the cache.

To run the unprofiled pair from a shell after installing the project, use:

```bash
beamform-lab benchmark --dataset picmus --mode numpy --runs 20 --warmup 2 \
  --save-image artifacts/exercise3-picmus-numpy-bmode.npy \
  --output results/exercise3-picmus-numpy.json

beamform-lab benchmark --dataset picmus --mode cupy --runs 20 --warmup 2 \
  --save-image artifacts/exercise3-picmus-cupy-bmode.npy \
  --output results/exercise3-picmus-cupy.json
```

## Nsight Systems trace

The notebook captures both modes with named NVTX ranges around dataset setup, initial CuPy upload, warmups, measured beamforming calls, and final output transfer. For a shell capture, use short runs and save each profile with a distinct prefix:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=artifacts/exercise3-picmus-numpy \
  beamform-lab benchmark --dataset picmus --mode numpy --runs 3 --warmup 1 --nvtx

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=artifacts/exercise3-picmus-cupy \
  beamform-lab benchmark --dataset picmus --mode cupy --runs 3 --warmup 1 --nvtx
```

In the visualizer, correlate the `measure/beamform/<mode>` host ranges with CUDA kernel and memory rows. Check whether the four NumPy-source HtoD transfers repeat inside each measured call and whether CuPy's input upload is confined to `setup/input_upload`. Inspect copy direction, byte count, kernel activity, and gaps. Use the unprofiled report for latency comparisons because tracing adds overhead.

## Success criteria

- Reports identify the UFF file, 75-transmit input, scan grid, source checksum, input-array byte sizes, and runtime.
- Peak-normalized NumPy and CuPy images are compared with an image plot and numerical error metrics.
- One-time upload/setup, repeated reconstruction percentiles, and final output transfer are presented separately.
- Traces reveal where input transfers occur and show the kernels launched for compounding.
- Conclusions are labeled as single-image reconstruction results for the recorded GPU/runtime, not streaming or live acquisition-to-display latency.

Dataset details and attribution requirements are documented in [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-picmus-experimental-resolutiondistortion-phantom).
