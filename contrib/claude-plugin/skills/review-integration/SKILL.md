---
name: review-integration
description: Review, audit, or check existing code that integrates Celeritas into a Geant4 application or experiment framework - TrackingManagerIntegration/SharedParams setup, SetupOptions, sensitive detectors, field, threading, CMake. Use when a user asks to find problems in such an integration or whether it is correct, rather than to write or change it.
---

Delegate the review to the `celeritas:integration-reviewer` agent instead of
reviewing the code yourself. Celeritas integration requirements change between
releases, so a review from memory reports problems that do not exist. For
example, Celeritas 0.7 tracking manager offload needs no run action, although
0.6 did.

## 1. Dispatch

Give the agent:
- The files or diff under review, including the `CMakeLists.txt` if there is
  one, since it usually states the required Celeritas version.
- The Celeritas and Geant4 versions, if known.

## 2. Report

Relay the agent's findings most severe first, each with its file and line, and
what it says passed or did not apply. Do not add Celeritas API requirements of
your own. If reading the code shows that a finding is wrong, drop it and say
so.

Leave the code unchanged unless the user asks for fixes. To make them, use the
`geant4-offload`, `framework-integration`, or `cmake-link` skill.
