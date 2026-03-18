---
name: commit-procedure
description: 'Pre-commit checklist and git commit formatting for the Celeritas project. Use when committing code, running pre-commit hooks, writing commit messages, checking tests before commit, or completing a task that requires a commit.'
---

# Commit Procedure

Execute these steps in order every time a task is complete and changes need to be committed.

## 1. Tests

Find the corresponding test file by mirroring the `src/` path and replacing `.hh`/`.cc` with `.test.cc`:

- `src/foo/Bar.cc` -> `test/foo/Bar.test.cc`

If you added or changed any public API -- including adding a method to an existing class -- add or update tests in that file. This applies to *all* changes, not just new classes.

> Note: object files and test names may differ from source paths.
> Example: `src/celeritas/ext/GeantImporter.cc` ->
> `src/celeritas/CMakeFiles/celeritas_geant4.dir/ext/GeantImporter.cc.o`
> and `celeritas/ext/GeantImporter.test.cc` -> `test/celeritas/ext_GeantImporter`.
> Use `ctest --show-only | grep <name>` to find the right test target.

## 2. Format

```bash
pre-commit run
git add <any files modified by pre-commit>
```

## 3. Compile

Confirm the build still succeeds before committing.

## 4. Commit

The commit message subject must use imperative mood ("Add X", "Fix Y", "Update Z" -- not "Added" or "Adds"). The body must include a quoted copy of the immediately preceding user prompt (excluding any metadata or file attachments -- plain text only):

```bash
git add <files>
pre-commit run        # Auto-formats code; re-add if it modifies files
git commit --trailer "Assisted-by: <agentic-tool> (<model-name>)" -m "Imperative action clause

Prompt: <verbatim user prompt text, wrapped in quotes>"
```

## Common Failure Modes

- Treating follow-up instructions within one feature as "incomplete" and deferring the commit indefinitely. Each self-contained feature or refactor warrants its own commit even if the user continues asking questions afterward.
- Skipping the test-file check because the change "only" added a method to an existing class rather than creating a new one. Always check.
