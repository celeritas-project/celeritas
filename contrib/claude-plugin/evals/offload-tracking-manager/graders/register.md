---
type: regex
pattern: 'RegisterPhysics\(\s*new\s+(celeritas::)?TrackingManagerConstructor'
target: { source: file, path: trackingmanager-offload.cc }
---
