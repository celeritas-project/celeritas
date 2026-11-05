---
type: regex
pattern: 'SetOptions\([\s\S]*?\)\s*;[\s\S]*?->Initialize\(\)'
target: { source: file, path: trackingmanager-offload.cc }
---
