---
description: Review an integration whose SetOptions call comes after Initialize()
tags: [fixture, review]
runs: 2
max_turns: 25
allowed_tools: [Read, Glob, Grep, Skill, Agent]
expected_outcome: The ordering bug is reported as the most severe problem
---

Review the Celeritas integration in `trackingmanager-offload.cc` for
problems. Do not edit any files.
