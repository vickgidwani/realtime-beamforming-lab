# Exercise 5: streaming overlap and tail latency

**Status:** The T4 rate sweep and the L4 optimization comparisons are complete. See the [companion notebook](05-streaming-tail-latency.ipynb) for the T4 experiments and the [L4 results below](#l4-optimization-results) for the setup carried into Exercise 7.

## Question

Can the GPU beamforming pipeline sustain a stated replay rate, and what happens to latency, queue depth, deadlines, and drops as offered load approaches or exceeds service capacity?

## Main comparison

The notebook and the [profiling scripts](../scripts) import the same [Python replay runner](../src/beamforming_lab/streaming_beamformer.py). The [Cython producer](../src/beamforming_lab/native_replay.pyx) is compiled on first use on Linux. Run the scripts from an environment with the lab installed, Cython and setuptools available, and a C compiler. The notebook holds the experiment matrices, plots, and interpretation.

The notebook replays 32 recorded PyMUST rotating-disk phantom acquisitions for 256 cycles (8,192 offered frames) per trial. Its 13-rate matrix compares serialized pinned-memory H2D transfer and beamforming with a two-slot path that uses asynchronous copy and compute streams. Both modes use the same precompiled Cython producer, drop-newest policy, and eight-frame waiting queue. The producer schedules against absolute monotonic deadlines in a `nogil` section; enqueueing each frame still needs the Python GIL.

Per-frame results include host-observed latency and queue wait, CUDA event timings for H2D and beamform/power, producer scheduling lateness, deadline misses, drops, and queue depth. The final power map is copied to the host once per trial. Nsight Systems should confirm stream overlap before describing the path as actual copy/compute overlap.

This is repeated replay of stored data, not live scanner acquisition. The dataset has no continuous live timestamps; the notebook generates the arrival schedule using the nominal 10 kHz PRF as a reference point.

## Main results

The Cython 200 Hz baseline completed all 8,192 frames on time in both modes. At 500 Hz both modes had deadline misses but no drops. At 600 Hz the serialized run also dropped 30 frames, while overlapped dropped none; drops first appeared at 700 Hz in the overlapped run. By 1 kHz, no accepted frame completed within one input period.

At 10 kHz, the 0.100 ms input period is much shorter than the GPU beamform-plus-power event (1.48–1.56 ms). The main run accepted 411 serialized frames and 354 overlapped frames; all accepted frames were late, so both modes had 100% total failure share. The Cython overlapped 10 kHz run had unusually high producer lateness, and the new B=1 batch replay provides a same-session check. Median H2D time (0.071–0.079 ms) was much smaller than GPU work. A larger queue would hold more stale work without increasing sustained service rate.

## Follow-up: microbatch beamforming at 10 kHz

To compare inline packing with concurrent host staging, run `scripts/benchmark_batched_beamforming.py --packer one_pass --batch-sizes 8 --modes async_explicit_stream` with and without `--concurrent-staging`. Use separate `--output` files. The staging worker draws from the same two pinned allocations, packs queued descriptors in order, and hands batches to the GPU dispatcher through a bounded queue. Host buffers return after H2D, while device slots stay occupied through reduction. The frame-descriptor queue still has its own capacity, so include staged and in-flight batches when reasoning about total buffered work. Add `--trace-stages` for Nsight Systems markers on both CPU threads.

With the [submission trace patch](../patches/README.md) installed, add `--trace-mach` to see output initialization and native binding time inside the Mach wrapper range. Use `--reuse-beamform-output` with `--modes async_explicit_stream` to allocate the output and build the fixed call arguments once per slot. This path clears the output on the compute stream before waiting for the input copy. Its compute event span starts after that clear, so compare H2D completion to kernel start on the GPU timeline.

Add `--prepare-mach` to the output-reuse condition to validate each device
slot once before replay. Both conditions still stage and transfer every
accepted batch from the host. With tracing enabled, preparation ranges move
outside the trial, and `mach/native_submit` remains in each batch.


The new section tests batch sizes B=1, 2, 4, 8, and 16 with the Cython producer and two-slot pipeline. It first checks Mach's batch axis offline against the B=1 power-Doppler reference, then runs the scheduled 8,192-frame replay. Reported outcomes include failure share, per-frame age, batch fill, host staging, H2D and GPU time, and image error when the full replay is accepted. Partial batches are zero-padded for the GPU call; only real acquisitions count toward frame metrics.

For a batch representing S input slots at requested rate f, the batch deadline is the scheduled time of its last represented slot plus S/f. Missing arrivals are drops; every accepted member of a late batch counts as late. The notebook also reports the stricter one-period-per-frame miss share and acquisition-to-output age, so the batch-aware deadline does not hide freshness costs.

The offline check preserved the full-ensemble image to relative RMSE <= 1.18e-7 and improved wall throughput from about 824 frames/s at B=1 to 5,479 at B=16. In the scheduled 10 kHz replay, B=8 peaked at about 1,463 accepted frames/s (2.4x the fresh B=1 control at 607 frames/s), but every offered frame still dropped or completed late. The B=8 p95 acquisition-to-output age was 12.59 ms; its batch budget was 0.8 ms versus a 1.81 ms median GPU event. No trial accepted the complete replay, so streaming image RMSE is unavailable. The T4 run wrote partial maps, but they are not included in the local results or Git commit because they cannot support a complete-ensemble accuracy check. The summary and plot are saved as `results/exercise5-batching-10khz.json` and `results/exercise5-batching-10khz.png`; detailed interpretation and tables are in the notebook.

Dataset details: [DATA_PROVENANCE.md](../DATA_PROVENANCE.md#active-dataset-pymust-rotating-disk-doppler-phantom).

## L4 optimization results

The replay repeats the same 32 PyMUST IQ frames from host RAM. Each accepted batch is packed into pinned RAM and copied to the L4. The geometry stays on the GPU. Unless noted, the comparisons below use a 251 × 251 scan grid, target batch size eight, a queue of eight waiting frame descriptors, two device slots, and 10,000 offered frames/s. Treat each row as its own comparison. Unprofiled L4 comparisons report medians across three 8,192-frame runs where available. Nsight captures diagnose the timeline and can change scheduling.

| Change or measurement | Result |
| --- | --- |
| Move the historical batch sweep from the Colab T4 to the GCE L4 | At batch eight with stock Mach, completed throughput rose from 1,463 to 1,976 frames/s and p95 frame age fell from 12.59 to 8.42 ms. Both hosts missed every offered-frame deadline. The hosts and runtimes differ. |
| Increase waiting capacity or add a second beamform compute stream | On the T4, a queue of 8,192 preserved the full map but drove p95 age to 4.04 s. A separate resident-input test improved from 2,815 to 2,840 frames/s with two compute streams. Keep one compute stream for the replay. |
| Patch Mach for explicit-stream async submission and trace the batch lifecycle | The initial paired 2,048-frame L4 trace completed 1,633 frames/s with a synchronous Mach call and 1,743 with the async call. The async trial had zero H2D copies overlapping a kernel. CPU staging took about 3.29 ms per batch and the next copy started a median 3.55 ms after the preceding kernel. |
| Replace frame-wise strided NumPy copies with a one-pass Cython gather | With inline async staging, median pack time fell from 3.00 to 0.58 ms and completed throughput rose from 2,109 to 6,203 frames/s. The traced gap before the next copy fell from 3.36 to 0.56 ms. H2D still did not overlap kernels in that capture. Packed buffers matched byte for byte, including partial and reordered batches. |
| Pack on a staging worker and return pinned buffers after `copy_done` | Against inline one-pass staging, completed throughput rose from 6,546 to 7,409 frames/s, and 28 of 186 H2D copies overlapped a kernel versus zero of 141. Median p95 frame age rose from 3.86 to 53.63 ms in that comparison. A device slot stays owned until `compute_done`. Packed batches can wait outside the descriptor queue. |
| Preallocate an output and prepare fixed launch arguments per device slot | Allocation originally took about 98 µs per call in the detailed trace. Reuse removed the allocation range and shortened the H2D-to-kernel gap from 310 to 244 µs in a separate trace. In a fresh unprofiled comparison, original allocation peaked at 8,025 frames/s versus 7,535 with reuse. Separate counter captures showed 41.3% and 43.3% mean SMs Active, respectively. These are whole-replay counters and differ from the T4's 39.9% per-kernel SM throughput. |
| Validate each slot once and retain the DLPack adapters | Against output reuse alone, completed throughput rose from 7,457 to 9,466 frames/s and median p95 frame age fell from 16.62 to 2.88 ms. Median H2D-to-beamform gap fell from 282 to 10 µs. The host typically entered the native launch API 99 µs before the copy finished. The native binding still consumes fresh DLPack capsules per launch. |

The Python replay runner and Linux Cython producer were extracted from the notebook so each condition can run from the same CLI. The [Mach patch stack](../patches/README.md) targets v0.2.1. During the detailed output-reuse trace, the input checks and preparation took about 192 µs of wall time and DLPack setup took 33 µs. The validate-once path moves both ranges before the trial. The nonprivileged CPU traces do not distinguish wrapper execution from time the dispatcher spent off CPU.

### Fastest measured setup

Use one explicit async compute stream, one-pass Cython packing, concurrent staging, per-slot output reuse, and prepared Mach launches. Keep two pinned host buffers, two device slots, queue capacity eight, and batch size eight. This is the fastest setup measured for completed throughput at 10 kHz. Each of the three unprofiled runs completed 9,420–9,646 frames/s. The best run completed 7,920 of 8,192 frames and dropped 272. The 200 Hz tests matched the reference power map within 1.3 × 10⁻⁷ relative RMSE at batch sizes seven and eight. At 10 kHz, every offered frame still dropped or missed both the batch-aware and strict frame deadlines. SM counters were collected before validate-once was added, so the fastest setup has no comparable SM counter measurement yet.

On the current L4 VM, select the installed prepared Mach overlay and run from the lab checkout:

```bash
PYTHONPATH="$(cat artifacts/mach-prepared-path.txt)" \
  .venv/bin/python scripts/benchmark_batched_beamforming.py \
  --modes async_explicit_stream --packer one_pass --concurrent-staging \
  --reuse-beamform-output --prepare-mach \
  --batch-sizes 8 --queue-capacity 8 --rate-hz 10000 --frames 8192 \
  --output artifacts/exercise5-best-replay.json
```

A replacement VM provisioned with `--patched-mach` installs the same stack under `artifacts/mach-async`. For a separate trace, add `--trace-stages --trace-mach`, reduce to 2,048 frames, and run under Nsight Systems. Raw L4 reports and traces live in the ignored `artifacts/` directory on the Mac and VM. See the [result index](../results/README.md#exercise-5-l4-artifacts) for their locations.
