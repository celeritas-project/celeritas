---
description: Write celer-sim JSON input that runs on the GPU with a field
tags: [smoke, standalone]
runs: 2
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill]
---

Write a `celer-sim` input file that runs 100 events, each with a single
1 GeV photon starting at the origin and moving along +z, in the geometry
`detector.gdml`, on the GPU, with a uniform 2 T magnetic field along z. Do not
write any files: print the complete JSON in your answer.
