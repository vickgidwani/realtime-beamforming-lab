# Exercise 5: streaming overlap and tail latency

**Status: notebook prepared; Colab T4 run pending.** See the [companion notebook](05-streaming-tail-latency.ipynb).

## Question

Can the GPU beamforming pipeline sustain a stated replay rate, and what happens to frame latency, queue depth, deadlines, and drops as offered load approaches or exceeds service capacity?

## Experiment

The notebook replays the 32 recorded PyMUST rotating-disk phantom acquisitions for 64 cycles per trial. It compares serialized pinned-memory H2D transfer and beamforming with a two-slot path that uses asynchronous copy and compute streams. A producer schedules arrivals at five rates from 1 to 15 kHz; the bounded queue holds eight waiting frames and drops the newest arrival when full. Setup and warmup are excluded from the measured trials.

Per-frame results include host-observed latency and queue wait, CUDA event time for H2D and beamforming plus power reduction, producer scheduling lateness, deadline misses, drops, and queue depth. The final power map is copied to the host once per trial. A one-ensemble reference checks the normalized map when no frames were dropped. Nsight Systems must confirm stream overlap before the result is described as actual copy/compute overlap.

This is repeated recorded-data replay, not live scanner acquisition. The file has no continuous live timestamps; its nominal 10 kHz PRF is used as a reference point, while the notebook generates and records its own arrival schedule.

## Run state

The notebook is prepared for the Colab T4 runtime. The next step is to connect the VS Code kernel, run the install/setup cells, and execute the sweep. The Nsight attach or launch command will be selected after checking which `nsys` version is present in that runtime.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
