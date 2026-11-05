# Celeritas plugin for Claude Code

This plugin helps people who *use* Celeritas: installing it, linking it from
CMake, adding GPU offload to Geant4 applications and experiment frameworks,
running the standalone apps, using it as a C++ library, tuning performance,
and diagnosing runs. It does not need a Celeritas source checkout. Contributors working on Celeritas
itself use the project skills in the repository's `.claude/skills/`.

## Install

```
/plugin marketplace add celeritas-project/celeritas
/plugin install celeritas@celeritas
```

To keep the plugin consistent with an installed Celeritas release, add the
marketplace at that release's tag instead:
`/plugin marketplace add celeritas-project/celeritas@vX.Y.Z`. Only releases
made after the plugin was added contain it. Without a pinned tag, the plugin
follows the default branch. Third-party marketplaces do not auto-update by
default: run `/plugin marketplace update celeritas` to pick up changes.

## Skills

| Skill | Use it to |
|-------|-----------|
| `/celeritas:doctor` | Detect the installed Celeritas, its build options (GPU, geometry, MPI, debug), Geant4, and GPU visibility |
| `/celeritas:install` | Choose and run an installation: Spack variants, CMake options and presets, compatibility constraints |
| `/celeritas:cmake-link` | Write or fix downstream `CMakeLists.txt` (`find_package`, `celeritas_target_link_libraries`, CUDA RDC) |
| `/celeritas:geant4-offload` | Add or adjust Celeritas offload in a Geant4 application (`SetupOptions`, SDs, fields, UI macros, validation) |
| `/celeritas:framework-integration` | Integrate with DD4hep/DDG4, LArSoft/art, or custom/task-based frameworks (Gaudi, CMSSW, Athena) |
| `/celeritas:new-offload-app` | Start a new Geant4+Celeritas app from the offload template matching the installed version |
| `/celeritas:standalone-input` | Write or fix input for `celer-sim`, `celer-optical`, `celer-g4`, `celer-geo`, `celer-export-geant` |
| `/celeritas:library-api` | Use Celeritas from C++ without Geant4 offload: programmatic transport, geometry, step data, custom GPU actions |
| `/celeritas:performance-tuning` | Estimate achievable speedup, size capacities, read timers, and profile |
| `/celeritas:troubleshoot` | Match Celeritas errors to fixes and summarize `celeritas.out.json` |

Claude also loads these skills automatically when a request matches their
description.

## Agent

`celeritas:integration-reviewer` is a read-only reviewer that audits
integration code (setup order, threading, options, sensitive detectors, MC
truth, CMake) and returns ranked findings. Ask Claude to "review this
Celeritas integration", or delegate to the agent by name.

## Design

- Skills give workflows and gotchas. They defer API details to the user's
  installed Celeritas headers and the versioned documentation at
  https://celeritas-project.github.io/celeritas/, so the advice matches the
  installed version.
- Helper scripts are read-only and use only the Python standard library.
- `scripts/ci/check-claude-plugin.py` in the Celeritas repository runs in
  pre-commit and fails when the plugin refers to files, headers, environment
  variables, `SetupOptions` fields, UI commands, or classes that no longer
  exist. It also checks the structure of the eval suite.

## Running evals

`evals/` holds a [`claude plugin eval`](https://code.claude.com/docs/en/plugin-evals)
suite. Each case runs Claude on a task and grades the result. By default
each case runs twice, once with the plugin and once without, and the report
shows the score difference (Δ). Every run is a real Claude session billed to
your credentials, so run the suite locally when skills change or before a
release, not on every commit. Run it from a Celeritas source checkout:
scaffolds copy fixtures from `example/geant4/`.

| Tier (tag) | Needs | Command |
|------------|-------|---------|
| `smoke` | Nothing extra | `claude plugin eval contrib/claude-plugin --tag smoke --no-publish` |
| `fixture` | `--scaffold`; `Edit` for edits; `Bash` (sandboxed) for the doctor case | `claude plugin eval contrib/claude-plugin --tag fixture --scaffold --allow-tools Edit Bash --no-publish` |
| `build` | Celeritas, Geant4, and CMake on `PATH`; a non-root user with the Bash sandbox (`bubblewrap`, `socat` on Linux) | `claude plugin eval contrib/claude-plugin --tag build --scaffold --allow-tools Edit Write Bash --no-publish` |

Useful options:
- `--runs 1` or `--ablation none` for a quick, cheaper check.
- `--max-cost-usd <n>` as a hard budget.
- `--model <id>` to pin the model when comparing runs over time.
- `--case <name>` to rerun one case.

With `--runs 1`, the smoke and fixture tiers each cost about $0.60 with
both arms (measured November 2026 with the default eval model). The default
of two runs per case roughly doubles that. Results go to `evals/results/`
(gitignored). The `build` case is the slowest: Claude
edits the example, builds it against the host install with the environment
the scaffold records in `build-env.sh`, and runs it. Eval runs do not inherit
your shell environment.

When adding a case, give it one of the tags above, a skill-fired indicator
(`tool_used` with `tool: Skill`), and at least one grader that a run without
the plugin could also pass, so Δ is meaningful.

## Roadmap

- `migrate-to-1.0`: help users move from `celeritas::SetupOptions` to
  `inp::FrameworkInput` once the integration classes accept it (planned for
  Celeritas 1.0).

See the user manual page for details:
https://celeritas-project.github.io/celeritas/user/usage/claude-plugin.html
