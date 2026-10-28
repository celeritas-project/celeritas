---
name: debug-test
description: Update .vscode/launch.json to debug a specific CTest test by name.
disable-model-invocation: true
---

Update `.vscode/launch.json` so its first debug configuration runs the CTest
test whose name (or substring) is `$ARGUMENTS`. If no name was given, ask for
one.

```bash
python3 scripts/dev/ctest-debug-launch.py "<test-name>"
```

If several tests match, the script errors and lists them: ask the user to
pick one and re-run with the more specific name.

If the test lives in a non-default build directory, pass it:

```bash
python3 scripts/dev/ctest-debug-launch.py --build-dir build-<preset> "<test-name>"
```

After the script succeeds, briefly confirm which test was selected.

(Mirrors `.github/prompts/debug-from-ctest.prompt.md`; keep the two in sync.)
