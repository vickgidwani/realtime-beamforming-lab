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

## Planned dataset: PICMUS experimental phantom

The intended next adapter is `PICMUS_experiment_resolution_distortion.uff`, a public experimental plane-wave dataset listed by the [USTB dataset catalog](https://unioslo.github.io/USTB/datasets.html). The USTB catalog says these UFF files contain channel data; the [mach compounding example](https://forest-neurotech.github.io/mach/examples/plane_wave_compound.html) demonstrates a 75-angle PICMUS workload. Before the adapter is merged, record the exact downloaded file's SHA-256, the UFF's embedded authorship/citation metadata, and the applicable dataset license. Do not bundle the raw UFF file in this repository.

## Provenance for new experiments

For every added dataset, record:

- stable dataset name, version/release, source URL, and retrieval date;
- whether it is simulated, phantom, or in-vivo, and whether it is raw RF, IQ, or already beamformed;
- probe, transmit sequence, dimensions, sample rate, and other parameters needed to reproduce beamforming;
- dataset-specific citation, license, and redistribution terms;
- raw-file checksum and the checksum of any transformed array used by the benchmark;
- each transformation, including dtype/layout changes, filters, compounding, and cropping.

Do not infer that a dataset is freely redistributable just because its download is public. The project code license does not grant rights to third-party data.
