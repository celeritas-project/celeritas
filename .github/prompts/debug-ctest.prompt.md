---
agent: 'agent'
description: Update .vscode/launch.json to debug a specific CTest test by name
tools: execute/getTerminalOutput, execute/runInTerminal
---

Update `.vscode/launch.json` so its first debug configuration runs the CTest
test whose name (or substring) the user provides.

Ask the user for the CTest test name if they haven't already provided one.
Then run:

```bash
python scripts/dev/debug-ctest.py "<test-name>"
```

If multiple tests match (the script will print an error listing them), ask the
user to disambiguate and re-run with the more specific name.

If a `--build-dir` override is needed (e.g. the test lives in a non-default
build directory), pass it:

```bash
python scripts/dev/debug-ctest.py --build-dir <dir> "<test-name>"
```

After the script succeeds, briefly confirm which test was selected.
