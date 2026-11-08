---
name: performance-tuning
description: Measure and improve Celeritas performance in Geant4 integrations and standalone runs - estimating achievable speedup, sizing track slots and buffers for CPU or GPU, reading Celeritas timers, profiling with Nsight Systems/Compute, rocprof, or Perfetto, slot diagnostics, and avoiding configurations that silently slow runs down. Use when a Celeritas run is slower than expected, runs out of GPU memory, or the user wants benchmark numbers.
---

Measure before changing anything, and change one setting at a time. Keep the
event sample, geometry, physics list, and thread count fixed between
comparisons.

## 1. Rule out slow configurations

Use `doctor` facts and the run log:

- **Debug build** (`celeritas.debug` not off): far slower by design. Rebuild
  in release mode before measuring.
- **Synchronizing timers:** `opts.action_times` (or `/celer/device/actionTimes`)
  synchronizes the GPU after every action. Disable it for throughput runs.
- **Wrong device:** GPU build running on CPU (`CELER_DISABLE_DEVICE`, no
  visible GPU), or a standalone input without `system.device`.
- **Architecture mismatch:** the "compute capability" warning at startup.
- **Too few track slots on GPU:** defaults sized for CPU, e.g. copied from the
  examples (`opts.max_num_tracks = 2024`).

## 2. Know the ceiling

Run the application three ways on the same events:

| Run | Setting | Measures |
|-----|---------|----------|
| Reference | `CELER_DISABLE=1` | Pure Geant4 time |
| Ceiling | `CELER_KILL_OFFLOAD=1` | Geant4 time with EM tracks killed: the best possible time with offload |
| Celeritas | (none) | Actual time with offload |

Speedup = reference / Celeritas. The ceiling run shows how much time remains
outside Celeritas (hadronics, the application's own code, I/O). If Celeritas
is already near the ceiling, tuning Celeritas will not help further.

## 3. Size capacities

All of these are **per stream** (per Geant4 worker thread) in `SetupOptions`.
In standalone input, `problem.control.capacity` holds the totals.

| Setting | CPU | GPU |
|---------|-----|-----|
| `opts.max_num_tracks` | ~1-4k per thread | As large as memory allows. Start around 2^18-2^20 in total, divided by the number of streams sharing the GPU. |
| `opts.initializer_capacity` | ~8-32x track slots | ~8x or more track slots (grows on demand) |
| `opts.auto_flush` | default | Larger values buffer more tracks per launch, at the cost of host memory and latency |
| `opts.secondary_stack_factor` | default (2) | Raise it if showers are dense (see the `geant4-offload` options reference) |
| `opts.track_order` | default (`none`) | Default `init_charge`. Try `reindex_*` orderings only with timing data. |

Many CPU threads feeding one GPU multiply per-stream memory. Prefer fewer
streams with more slots each. Then check GPU memory with `nvidia-smi` or
`rocm-smi` while running.

## 4. Find where time goes

- **Celeritas timers:** with diagnostics output enabled, `result.time` in
  `celeritas.out.json` holds setup time, per-event times, and (if enabled)
  per-action times. Summarize it with the `troubleshoot` skill's script. Turn
  per-action timing on only for diagnosis.
- **Timelines:** set `CELER_ENABLE_PROFILING=1` and run under a profiler. No
  code changes are needed for NVTX/ROCTX.

  ```bash
  CELER_ENABLE_PROFILING=1 nsys profile --trace=cuda,nvtx,osrt \
    --nvtx-capture=celeritas -o report -f true ./myapp run.mac
  CELER_ENABLE_PROFILING=1 rocprof --roctx-trace --hip-trace ./myapp run.mac
  ```

  Celeritas annotations are in the `celeritas` domain/category, so they can
  be filtered alongside the application's own ranges. Counters (active/alive
  tracks, hits) are available with NVTX and Perfetto.
- **CPU timelines:** Perfetto, in CPU-only builds. Standalone apps use
  `problem.diagnostics.perfetto_file`. In applications, create a
  `celeritas::TracingSession` (`corecel/sys/TracingSession.hh`).
- **Kernels:** Nsight Compute (`ncu --set full --kernel-name launch_action_impl ...`).
  Source-line mapping requires Celeritas built with `-lineinfo` in
  `CMAKE_CUDA_FLAGS`.
- **Slot occupancy:** `opts.slot_diagnostic_prefix` writes per-slot activity.
  Plot it with `scripts/user/plot-slot-diagnostic.py` from the Celeritas
  source to see whether slots sit empty (a tail effect: consider larger
  `auto_flush` or more events in flight).

## 5. Typical findings

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| GPU mostly idle in the timeline | Geant4 (CPU) dominates; too few tracks per flush | Check the ceiling; raise `auto_flush` and track slots |
| Long tail of steps with few active tracks | Low-energy or looping tracks | Check `opts.max_steps`; review production cuts |
| Setup takes minutes | Geometry conversion and physics table import | One-time cost per job. Amortize over more events per job. |
| Out of GPU memory | Too many streams times slots | Reduce per-stream slots or the stream count |
| Results change with slot counts | Expected: random-number streams can depend on track slots (the `CELERITAS_RESEED=track` build option improves reproducibility) | Compare distributions, not events |

## References

- Profiling manual: https://celeritas-project.github.io/celeritas/user/usage/execution/profiling.html
- Environment variables: https://celeritas-project.github.io/celeritas/user/usage/execution/environment.html
