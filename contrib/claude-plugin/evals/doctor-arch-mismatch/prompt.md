---
description: Detect a GPU architecture mismatch with the doctor probe
tags: [fixture, doctor]
runs: 2
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Bash]
expected_outcome: Reports capability 80 vs compiled architecture 70 and recommends rebuilding
---

Celeritas is installed under `./fake-install` in this directory. Check
whether this installation is usable on this machine's GPU and report any
problems. Do not modify anything.
