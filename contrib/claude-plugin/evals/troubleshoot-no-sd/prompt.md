---
description: Explain a fatal 'no G4 sensitive detectors' setup error
tags: [smoke, troubleshoot]
runs: 2
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill]
---

My Geant4 application with Celeritas offload (tracking manager integration,
Celeritas 0.7) aborts at the start of the run with:

```
-------- EEEE ------- G4Exception-START -------- EEEE -------
*** G4Exception : celer.setup
      issued by : celeritas/ext/GeantSd.cc:226
Celeritas runtime error: no G4 sensitive detectors are defined: set `SetupOptions.sd.enabled` to `false` if this is expected
*** Fatal Exception *** core dump ***
```

What is going on and how do I fix it? Show the code change.
