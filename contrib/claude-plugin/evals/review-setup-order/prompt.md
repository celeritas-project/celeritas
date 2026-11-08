---
description: Review an integration with a setup order bug and a track ID assumption
tags: [fixture, review]
runs: 2
max_turns: 25
allowed_tools: [Read, Glob, Grep, Skill, Agent]
expected_outcome: Both bugs are reported, with no false claim that a run action is required
---

Review the Celeritas integration in `trackingmanager-offload.cc` for
problems. Do not edit any files.
