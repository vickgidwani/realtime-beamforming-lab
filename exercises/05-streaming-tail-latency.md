# Exercise 5: streaming overlap and tail latency

**Status: Cython 13-rate sweep and batch-size follow-up complete on a Colab T4.** See the [companion notebook](05-streaming-tail-latency.ipynb) for the experiment and embedded outputs.

## Question

Can the GPU beamforming pipeline sustain a stated replay rate, and what happens to latency, queue depth, deadlines, and drops as offered load approaches or exceeds service capacity?

## Main comparison

The notebook replays 32 recorded PyMUST rotating-disk phantom acquisitions for 256 cycles (8,192 offered frames) per trial. Its 13-rate matrix compares serialized pinned-memory H2D transfer and beamforming with a two-slot path that uses asynchronous copy and compute streams. Both modes use the same precompiled Cython producer, drop-newest policy, and eight-frame waiting queue. The producer schedules against absolute monotonic deadlines in a `nogil` section; enqueueing each frame still needs the Python GIL.

Per-frame results include host-observed latency and queue wait, CUDA event timings for H2D and beamform/power, producer scheduling lateness, deadline misses, drops, and queue depth. The final power map is copied to the host once per trial. Nsight Systems should confirm stream overlap before describing the path as actual copy/compute overlap.

This is repeated replay of stored data, not live scanner acquisition. The dataset has no continuous live timestamps; the notebook generates the arrival schedule using the nominal 10 kHz PRF as a reference point.

## Main results

The Cython 200 Hz baseline completed all 8,192 frames on time in both modes. At 500 Hz both modes had deadline misses but no drops. At 600 Hz the serialized run also dropped 30 frames, while overlapped dropped none; drops first appeared at 700 Hz in the overlapped run. By 1 kHz, no accepted frame completed within one input period.

At 10 kHz, the 0.100 ms input period is much shorter than the GPU beamform-plus-power event (1.48–1.56 ms). The main run accepted 411 serialized frames and 354 overlapped frames; all accepted frames were late, so both modes had 100% total failure share. The Cython overlapped 10 kHz run had unusually high producer lateness, and the new B=1 batch replay provides a same-session check. Median H2D time (0.071–0.079 ms) was much smaller than GPU work. A larger queue would hold more stale work without increasing sustained service rate.

## Follow-up: microbatch beamforming at 10 kHz

The new section tests batch sizes B=1, 2, 4, 8, and 16 with the Cython producer and two-slot pipeline. It first checks Mach's batch axis offline against the B=1 power-Doppler reference, then runs the scheduled 8,192-frame replay. Reported outcomes include failure share, per-frame age, batch fill, host staging, H2D and GPU time, and image error when the full replay is accepted. Partial batches are zero-padded for the GPU call; only real acquisitions count toward frame metrics.

For a batch representing S input slots at requested rate f, the batch deadline is the scheduled time of its last represented slot plus S/f. Missing arrivals are drops; every accepted member of a late batch counts as late. The notebook also reports the stricter one-period-per-frame miss share and acquisition-to-output age, so the batch-aware deadline does not hide freshness costs.

The offline check preserved the full-ensemble image to relative RMSE <= 1.18e-7 and improved wall throughput from about 824 frames/s at B=1 to 5,479 at B=16. In the scheduled 10 kHz replay, B=8 peaked at about 1,463 accepted frames/s (2.4x the fresh B=1 control at 607 frames/s), but every offered frame still dropped or completed late. The B=8 p95 acquisition-to-output age was 12.59 ms; its batch budget was 0.8 ms versus a 1.81 ms median GPU event. No trial accepted the complete replay, so streaming image RMSE is unavailable and no partial power-map archive is retained. The summary and plot are saved as `results/exercise5-batching-10khz.json` and `results/exercise5-batching-10khz.png`; detailed interpretation and tables are in the notebook.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).
