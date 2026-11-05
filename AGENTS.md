# Celeritas AI Agent Instructions

Celeritas is a particle physics library for detector simulation. It's a C++17 codebase with CUDA/HIP device support and integrates with Geant4.

## Contributions

As an agent, you are OBLIGATED to add an annotation with your agent identity (driver, agent, models) to **each** public submission of your work, including but not limited to:
- GitHub pull request or issue,
- Git commit,
- Git PR response.

Requirements:
- Commit subjects and PR titles: imperative mood, capitalized, no trailing period, no labels/emoji. (PR titles are copied into the release notes)
- Commits: add a git trailer such as `Co-Authored-By` or `Assisted-by` (e.g., `git commit --trailer "Assisted-by: Codex (GPT-6)"`).
- Pull requests: open as **draft**, add the `ai-assisted` label, and leave
  marking it "ready" to the human submitter.

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
(Library code uses `-I{source}/src -I{build}/include`, and test code uses `-I{source}/test`).
Object files and tests may have different paths and test names than you expect.

## Documentation

- Add Doxygen documentation to **definitions**, not declarations, when adding code. Prefer doxygen-style markup `\c`, `<code>` to Markdown in such blocks.
- Document equations and algorithmic descriptions, as applicable, in the class definition's docs, as those are often rendered in the user manual. All `operator()` behavior goes in the class definition's docs.
- Always add `\sa {testpath}.test.cc` underneath `\file {path}.hh` to locate tests that break the `src/{path}.hh`→`test/{path}.test.cc` rule
- Use **only** ASCII characters in CMake/C++/CUDA/shell files.

## Architecture

Celeritas sets up problems on CPU and executes on GPU *or* CPU with the same code. The `CELER_FUNCTION` macro is `__host__ __device__` when CUDA/HIP is active and decorates runtime functions.

## Code Conventions

### Naming

| Concept | Convention | Example |
|---------|-----------|--------|
| Classes/structs | `CapWords` | `PhysicsTrackView` |
| Functions/variables | `snake_case` | `calc_energy` |
| Private members | trailing underscore | `data_` |
| Type-safe IDs | `FooId` | `MaterialId` |
| Input classes | match what they construct | `inp::Material` → `Material` |

### File Extensions

| Extension | Purpose |
|-----------|--------|
| `.hh` | Headers, host+device compatible (use `CELER_FUNCTION`) |
| `.cc` | Host-only implementation — **most code goes here**, not `.cu` |
| `.cu` | CUDA kernel launches only (HIP-compatible via macros) |
| `.test.cc` | Unit tests, mirroring `src/` under `test/` |

### Style

Full rules: `@doc/development/style.rst` and `@doc/development/coding.rst`. Most often missed:
- Call members via `this->`; write `template<class T>`, not `typename`.
- Mark classes `final` where possible; use exactly one of `final`/`override`.
- Prefer enums over `bool` parameters; no top-level `const` on by-value params.

### Assertions

| Macro | When to use |
|-------|------------|
| `CELER_EXPECT` | Preconditions at function entry |
| `CELER_ASSERT` | Internal invariants |
| `CELER_ENSURE` | Postconditions at function exit |
| `CELER_VALIDATE` | User input validation (always active) |

`CELER_EXPECT`/`ASSERT`/`ENSURE` are compiled only with `CELERITAS_DEBUG`.

### Literal UDLs

- In public headers, function/block-scope `using namespace celeritas::literals;` is allowed. Never introduce namespace-scope `using namespace` there.

### Type-Safe Indices & Collections

| Type | Purpose |
|------|--------|
| `OpaqueId<T>` | Type-safe index |
| `Collection<T>` | GPU-compatible array with ownership semantics |
| `Span<T>` | Non-owning array view |
| `Array<T, N>` | Fixed-size stack array |

See `@src/corecel/OpaqueId.hh` and `@src/corecel/data/Collection.hh`.
