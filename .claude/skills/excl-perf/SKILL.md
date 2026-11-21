---
name: excl-perf
description: Build Celeritas in the ExCL apptainer container and benchmark celer-optical on the ExCL GPUs (presets, architectures, old-commit and single-precision build workarounds, output/counter/kernel fields). Use for claude-excl builds or when reproducing/extending dune-perf timing results.
---

Celeritas-side knowledge for the ExCL workflow. The run/analysis half lives in the **dune-perf** repo at
`/work/run/dune-perf` (skill: `/work/run/dune-perf/.claude/skills/celer-optical-perf/SKILL.md`; read it
before running or analyzing sweeps). Both repos must be available locally; this checkout is `/work/src`
(branch `excl-env`, pushes go only to the `bot` remote, the user opens PRs).

## Layout and environment (apptainer session, launcher `claude-excl-celer.sh`)
- Source `/work/src`; builds `/work/build` (local scratch); installs `/work/install` (networked); `/work` is
  read-only otherwise. `TMPDIR`, `XDG_CACHE_HOME`, `CCACHE_DIR` come from the launcher.
- Environment: first source one of `/auto/projects/celeritas/environments/scripts/dunesw-{cuda,hip,nodevice}-env.sh`,
  then `. scripts/env/claude-excl.sh` (gcc-13, `CUDAARCHS="70;80;90;120"`, `CELER_DISABLE_DEVICE=1`, ccache vars).
- Presets in `scripts/cmake-presets/claude-excl.json` (include it from the gitignored `CMakeUserPresets.json`):
  `release-orange-{cuda,hip,nodevice}`; CUDA arches 70;80;90;120 (V100, A100, H100, RTX 5090), HIP arches
  gfx906;gfx908;gfx942 (MI60, MI100, MI300A). OpenMP (`track`) only on cuda/hip. Installs are relocatable
  (`CMAKE_INSTALL_RPATH=$ORIGIN:$ORIGIN/../lib`), so an install made in the container runs on the nodes.
- `-j`: `squeue --me -h -w $(hostname -s) -o %C` each time (never `nproc`, never a remembered number).
- The container has no GPU; GPU runs go through Slurm on ExCL nodes (see the dune-perf skill for node map).
- After benchmark runs, do not delete build/install trees or worktrees; report the new ones instead.

## Building old commits / variants
- Old commits have no presets: use `/work/run/dune-perf/celer-optical/build.sh` with `--preset-source /work/src`
  (resolves the claude-excl cache variables into plain `-D` options, drops `-Werror`). Use a dedicated worktree
  under `/work/build`, never `/work/src`.
- 1e512029 ROCm needs `-D__HIP_PLATFORM_AMD__=1` (covfie headers are included before `corecel/DeviceRuntimeApi.hh`).
- Single precision: `-DCELERITAS_REAL_TYPE=float`, plus `-DCELERITAS_USE_LArSoft=OFF` because
  `src/larceler` (`LarStandaloneRunner.cc`, `PDFullSimCeler.cc`) assigns `Array<double,3>` to float `Real3`
  on e0fd574f. celer-optical does not use it. Use a separate build/install dir per precision.
- Check `build.config.real_type` and `gpu_architectures` in the output JSON to confirm what was run.

## celer-optical input/output facts
- Inputs (in dune-perf): `dune-zwires/run-gpu.inp.json` (z-only wire model) and `dune10kt/primary-gpu.inp.json`
  (DUNE 10kt); 67,108,864 isotropic primaries, ORANGE geometry. Paths are relative to the run directory.
- `problem.timers.action` adds a device sync per action; `problem.timers.step` a sync per step. Turning the
  action timer off changed throughput only 0-3% (single runs).
- Output `result.time`: `total` (transport only), `setup` (not in total), `actions`, `steps`;
  `result.counters`: `steps`, `step_iters`, `num_errored`, `num_cut`, generator `num_generated`.
  `system.device` / `system.kernels`: multiprocessors, clocks, max threads per CU, per-kernel registers,
  local memory, blocks per CU and occupancy (use `along-step-propagate`; it is register-limited).
- Throughput = photons / `time.total`; startup is excluded. Runtime should track `counters.steps`/`step_iters`
  (develop vs 1e512029 differ by ~0.01%); the dune10kt speedup from 1e512029 to develop was ~10-19x.
- Device-property quirks: AMD constant-memory fields are sentinels, AMD `capability` is 9.0 for both gfx906
  and gfx908, RTX 5090 has 1536 max threads/SM (not 2048), which raises occupancy % at equal register use.

## Access
- AMD discrete GPUs (explorer, cousteau, maxwell) need the `render` group for `/dev/kfd`; check `id -nG`
  inside a Slurm job if HIP reports no device.
