# Celeritas plugin for Claude Code

This plugin helps people who *use* Celeritas: adding GPU offload to a Geant4
application, linking Celeritas from CMake, and diagnosing Celeritas runs. It
does not need a Celeritas source checkout. Contributors working on Celeritas
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
| `/celeritas:cmake-link` | Write or fix downstream `CMakeLists.txt` (`find_package`, `celeritas_target_link_libraries`, CUDA RDC) |
| `/celeritas:geant4-offload` | Add or adjust Celeritas offload in a Geant4 application (`SetupOptions`, SDs, fields, UI macros, validation) |
| `/celeritas:troubleshoot` | Match Celeritas errors to fixes and summarize `celeritas.out.json` |

Claude also loads these skills automatically when a request matches their
description.

## Design

- Skills give workflows and gotchas. They defer API details to the user's
  installed Celeritas headers and the versioned documentation at
  https://celeritas-project.github.io/celeritas/, so the advice matches the
  installed version.
- Helper scripts are read-only and use only the Python standard library.
- `scripts/ci/check-claude-plugin.py` in the Celeritas repository runs in
  pre-commit and fails when the plugin refers to files, environment
  variables, `SetupOptions` fields, UI commands, or classes that no longer
  exist.

See the user manual page for details:
https://celeritas-project.github.io/celeritas/user/usage/claude-plugin.html
