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
