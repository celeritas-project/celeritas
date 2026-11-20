# Celeritas AI Agent Instructions

Celeritas is a particle physics library for detector simulation. The codebase uses C++17 with CUDA/HIP device support and typically integrates with Geant4.

## Contributions

You MUST annotate contributions with your identity, even when acting on behalf of a human.
- A contribution is a commit, pull request, issue, comment, or any other work
  with human-readable text.
- Your identity is the agent, assistant, or harness; add the model, if known.
- Annotations should be in a body or description, not in subjects or titles.
- "Body" parameters submitted to GitHub API endpoints must contain a 🤖 emoji.

Submission requirements:
- Issues: always read and follow the
  `issue` skill (`.claude/skills/issue/SKILL.md`) to submit an issue.
- Pull requests: always follow the `pull-request` skill
  (`.claude/skills/pull-request/SKILL.md`) to submit a pull request.
  The description must contain a 🤖 and a "Generated with" statement (the PR
  workflow then adds `ai-generated`).
- Comments: *every* comment on an issue or pull request (including review
  replies) *must* disclose the agent harness and model in clear text and
  include a 🤖 emoji, e.g. `🤖 (Claude Code, Sonnet 5.5)`.
- Credit: `ai-generated` submissions should *not* modify
  `scripts/release/contributors.jsonl`; only human-authored or `ai-assisted`
  work is automatically eligible.
  Never add an entry for an agent or for purely generated work.

Commit requirements:
- Add a git trailer such as `Co-Authored-By` or `Assisted-By`
  (e.g., `git commit --trailer "Assisted-by: Codex (GPT-6)"`).
- Commit subjects and PR titles: imperative mood, capitalized, no trailing
  period, no labels/emoji.
  (PR titles are copied into the release notes.)
- History: you may force-push a PR branch *only* when it is in draft
  form and not under review.
  Local history modification is allowed unless it diverges from the remote
  branch.
  Push new commits rather than amending remote commits, and incorporate
  upstream changes by *merging* `upstream/develop` (see `CONTRIBUTING.rst`).

Prohibitions:
- Never submit a pull request without `draft=True`.
- Never mark a pull request as ready.
- Never mark a review comment as resolved.
- Never rebase or force-push a pull request branch that a user marked as ready.

## File Organization

See `@README.md#Directory structure`

### Libraries

- `corecel/`: GPU abstractions, data structures, utilities
- `geocel/`: Geometry interfaces (ORANGE, VecGeom, Geant4)
- `orange/`: Native Celeritas geometry engine
- `celeritas/`: Physics (EM processes, particles, materials)
- `accel/`: Geant4 integration layer
- `larceler,ddceler`: special plugin integrations (possibly disabled)

Libraries depend strictly downward:
`corecel` → `geocel` → `orange` → `celeritas` → `accel`.
Optional-dependency code is compiled into separate targets (e.g.
`src/celeritas/ext/` → `celeritas_geant4`) and guarded by
`CELERITAS_USE_<Pkg>` macros from the generated `corecel/Config.hh`.

Includes for library and test code are relative to the library and test directories.
(Library code uses `-I{source}/src -I{build}/include`, and test code also uses `-I{source}/test`).
Object files and binaries may have different paths and names than their source tree locations.

## Documentation

- Add Doxygen documentation to *definitions*, never on *declarations*. Prefer Doxygen-style markup `\c`, `<code>` to Markdown in such blocks.
- Document equations and algorithmic descriptions, as applicable, in the class definition's docs, as those are often rendered in the user manual. All `operator()` behavior goes in the class definition's docs.
- Always add `\sa {testpath}.test.cc` underneath `\file {path}.hh` to locate tests that break the `src/{path}.hh`→`test/{path}.test.cc` rule.
- Use **only** ASCII characters in CMake/C++/CUDA/shell files.

## Architecture

Celeritas sets up problems on CPU and executes on GPU *or* CPU with the same code. Most testing is CPU-only. The `CELER_FUNCTION` macro is `__host__ __device__` when CUDA/HIP is active and decorates runtime functions.

## Code Conventions

Full rules: `@doc/development/style.rst` and `@doc/development/coding.rst`

| Concept | Convention | Example |
|---------|-----------|--------|
| Classes/structs | `CapWords` | `PhysicsTrackView` |
| Functions/variables | `snake_case` | `calc_energy` |
| Private members | trailing underscore | `data_` |
| Type-safe indices (`OpaqueId`) | `FooId` | `MaterialId` |

### File Extensions

| Extension | Purpose |
|-----------|--------|
| `.hh` | Headers, host+device compatible (use `CELER_FUNCTION`) |
| `.cc` | Host-only implementation — **most code goes here**, not `.cu` |
| `.cu` | CUDA kernel launches only (HIP-compatible via macros) |
| `.test.cc` | Unit tests, mirroring `src/` under `test/` |

### Style

- Avoid writing long functions (over ~50 lines): break into focused helper
  functions/classes with descriptive names.
- Avoid repeating long blocks of common code across files: refactor to a common
  detail pattern (see `@src/AGENTS.md`).
- Always call member functions via `this->`.
- Write `template<class T>`, not `<typename T>`.
- Mark classes `final` where possible; use exactly one of `final`/`override`.
- Prefer enums over `bool` parameters; avoid top-level `const` on by-value params.
