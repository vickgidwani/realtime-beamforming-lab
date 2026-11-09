# Data provenance

This file records where benchmark inputs come from, what they represent, and how the project transforms them. Raw datasets are downloaded or supplied by the user; they are not checked into this repository.

## Active dataset: PyMUST rotating-disk Doppler phantom

| Field | Value |
| --- | --- |
| Upstream file | `PWI_disk.mat` |
| Source | [PyMUST example data at a pinned upstream revision](https://github.com/creatis-ULTIM/PyMUST/blob/fe73cc3911a51b48b36095c2e0ad29c266532447/examples/data/PWI_disk.mat) |
| Loader | `mach.io.must.download_pymust_doppler_data()`; the helper downloads and caches the file |
| Raw file fingerprint | 2,085,623 bytes; SHA-256 `c349dc1d677c561434fd0e4a74142c4a0a44b7e6ae0a42d446c078a528ef58c1` (upstream loader's expected integrity hash) |
| Data type | Raw ultrasound RF channel data from a tissue-mimicking phantom with a rotating disk |
| Acquisition described by upstream example | 128-element linear array; 32 plane-wave acquisitions; 334 fast-time samples; 6.7 MHz sample rate; 5 MHz center frequency; 1480 m/s speed of sound; 10 kHz PRF |
| Dataset citation | Garcia, D. “Make the most of MUST, an open-source MATLAB UltraSound Toolbox,” 2021 IEEE International Ultrasonics Symposium, DOI: [10.1109/IUS52206.2021.9593605](https://doi.org/10.1109/IUS52206.2021.9593605). See also the [MUST site](https://www.biomecardio.com/MUST/index.html) and the [mach Doppler example](https://forest-neurotech.github.io/mach/examples/doppler.html). |
| License | The repository's MIT license covers its code, not this data. PyMUST's source repository is LGPL-3.0; the raw example file's separate redistribution terms are not established here. The raw file is not redistributed; verify upstream terms before re-hosting it. |

This is measured phantom data, not simulated data and not a brain/fUSI recording. The controlled rotating disk produces Doppler shifts useful for checking the processing pipeline.

### Transformations applied

1. Load the `RF` array from `PWI_disk.mat`.
2. Convert RF to complex IQ using `pymust.rf2iq` and the acquisition parameters shipped with the data.
3. Reorder from `(samples, elements, frames)` to contiguous `(elements, samples, frames)` and cast to `complex64` for `mach`.
4. Construct probe coordinates, the 2D scan grid, and plane-wave transmit arrival times from the recorded acquisition parameters.
5. Beamform with `mach`, then sum squared complex magnitude across the ensemble to form the Power Doppler map.

### Fingerprinting

Each benchmark report stores the RF-array shape, dtype, and a SHA-256 fingerprint. The fingerprint is calculated from the loaded array's dtype string, its shape encoded as little-endian `int64`, and its bytes in C order. This identifies the array content used by the run; it is **not** a checksum of the original MAT container file.

## Active dataset: PICMUS experimental resolution/distortion phantom

| Field | Value |
| --- | --- |
| Upstream file | `PICMUS_experiment_resolution_distortion.uff` |
| Source | [USTB dataset catalog / Zenodo file](https://unioslo.github.io/USTB/datasets.html) |
| Loader | `mach.io.utils.cached_download()` followed by `pyuff_ustb.Uff` and `mach.io.uff.create_beamforming_setup()` |
| Raw file fingerprint | 145,518,524 bytes; SHA-256 `c93af0781daeebf771e53a42a629d4f311407410166ef2f1d227e9d2a1b8c641` (verified by the pinned mach example's downloader) |
| Data type | Experimental phantom ultrasound channel data in UFF format |
| Acquisition described by mach example | 128-element linear array; 75 plane-wave transmissions from −16° to +16°; 3,328 samples per acquisition; one frame; 20.8 MHz sampling frequency |
| Transmit frequency | 5.208 MHz, from the [PICMUS transmit-pulse specification](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/transmit_pulse); the challenge says this was the nearest Verasonics Vantage setting to the roughly 5.13 MHz target |
| UFF pulse metadata | `channel_data.pulse` is absent in this file; `modulation_frequency = 0` marks raw RF for Mach’s Hilbert path and is not the transmit frequency |
| Imaging grid | 387 lateral × 609 depth points in the example setup |
| Dataset citation | The UFF embeds dataset authorship metadata. The loader captures `channel_data.print_authorship()` output in benchmark provenance when the installed reader exposes it. Follow the attribution displayed in the file and the [PICMUS challenge](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) requirements. |
| License | The raw UFF file is cached locally and is not redistributed by this repository. Follow the dataset's published reuse terms; the repository's MIT license covers code only. |

This dataset is a larger, multi-angle image-quality workload than the rotating-disk phantom. It is suitable for testing per-angle beamforming, coherent compounding, and data movement with a denser scan grid. It is not an in-vivo brain recording and does not provide a slow-time hemodynamic sequence for functional analysis.

### Transformations applied

1. Download and verify the UFF file by expected size and SHA-256.
2. Read `/channel_data` and `/scan` from the UFF container.
3. Use mach's UFF adapter to prepare contiguous channel arrays, probe and scan geometry, and transmit arrival times.
4. Coherently beamform and compound all 75 plane-wave transmissions with `mach.experimental.beamform()`.
5. Convert the compounded complex image to magnitude for the saved B-mode array.

The `picmus` benchmark records the time to load/prepare the dataset separately from the repeated reconstruction latency. In CuPy mode, channel and geometry arrays are uploaded once and reused. In NumPy mode, mach receives host arrays for each repeated reconstruction. The measured deadline is a reconstruction-only proxy because this file contains one pre-acquired frame, not a live stream.

## Provenance for new experiments

For every added dataset, record:

- stable dataset name, version/release, source URL, and retrieval date;
- whether it is simulated, phantom, or in-vivo, and whether it is raw RF, IQ, or already beamformed;
- probe, transmit sequence, dimensions, sample rate, and other parameters needed to reproduce beamforming;
- dataset-specific citation, license, and redistribution terms;
- raw-file checksum and the checksum of any transformed array used by the benchmark;
- each transformation, including dtype/layout changes, filters, compounding, and cropping.

Do not infer that a dataset is freely redistributable just because its download is public. The project code license does not grant rights to third-party data.
