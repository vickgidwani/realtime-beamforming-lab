# Recording results

Keep small JSON summaries with the experiment notes. Do not commit raw ultrasound files, large benchmark arrays, or large profiler traces unless their license and size make that appropriate. Those files are ignored by Git; preserve them elsewhere and include a path or artifact link in the result note when useful.

For each comparison, record:

- repository commit, dataset ID/version, raw-file or RF-array fingerprint, transmit/frame dimensions, and transformations;
- GPU model, compute capability, driver, CUDA runtime, CPU, operating system, and package versions;
- exact command, mode, batch size (when applicable), transmit count, scan grid, warmup count, and run count;
- p50/p95/p99/max latency, throughput, deadline miss rate, transfer bytes, and dataset-appropriate correctness error (Power Doppler map or compounded image);
- profiler version and trace filename, if profiled;
- whether the run used recorded data, repeated frames, or synthetic data.

Benchmark reports automatically include dataset and runtime provenance where the environment exposes it. A report without a GPU field or Git revision should be treated as incomplete and annotated manually before comparing it with another run.

## Exercise 5 L4 artifacts

The [recorded comparison and fastest measured configuration](../exercises/05-streaming-tail-latency.md#l4-optimization-results) summarize the local artifacts. The following paths are relative to the lab's ignored `artifacts/` directory on the Mac and VM.

| Experiment | Directory |
| --- | --- |
| Historical T4 configuration replayed on the L4 | `l4-exercise5/` |
| Async baseline and detailed batch NVTX timeline | `exercise5-l4-trace/`, `exercise5-nvtx/` |
| Strided versus one-pass packing | `exercise5-packing/` |
| Inline versus concurrent staging | `exercise5-concurrent-staging/` |
| Output reuse and Mach wrapper breakdown | `exercise5-submit/`, `exercise5-wrapper-nvtx/` |
| Unprofiled throughput and separate L4 SM counter captures | `exercise5-throughput-sm/` |
| Validate-once comparison, parity checks, and traces | `exercise5-prepared/` |

## Exercise 7 L4 artifacts

The [1 kHz baseline and replay command](../exercises/07-gpu-resident-decoder.md#run-the-first-l4-replay) describe the saved files under `artifacts/exercise7/`. The baseline JSON records timing and deadline decisions per window, and the adjacent NPZ holds the embeddings keyed by window ID. The counterfactual runs check missing frames and late delivery.

| Artifact | Filename |
| --- | --- |
| Encoder and preprocessing check | `resnet50-embedding.onnx`, `preprocess-smoke.json`, `check_preprocess.py` |
| Unprofiled 1 kHz replay and embeddings | `replay-1khz.json`, `replay-1khz.npz` |
| GPU handoff trace and SQL export | `replay-1khz-trace.nsys-rep`, `replay-1khz-trace.sqlite` |
| Missing-frame and late-output checks | `replay-missing17.*`, `replay-missing31.*`, `replay-late.*` |

The Mac copy of the lab also has `artifacts/l4-archive/l4-artifact-manifest.json`, with SHA-256 checksums and local paths for the VM artifacts preserved before shutdown. The source and patch stack live in the lab checkout. The model, traces, and reports remain in Git-ignored artifact directories.
