---
type: llm
---

The sensitive detector's `ProcessHits` stores energy in `edep_by_track_`, a
`std::vector` indexed by `step->GetTrack()->GetTrackID()` and resized to
`index + 1`. Tracks created inside Celeritas have IDs counting down from
`INT_MAX`, so the resize attempts to allocate about two billion elements.

PASS if the response flags this vector and gives that reason: tracks created
by Celeritas (such as its secondaries) have very large track IDs, near
`INT_MAX`.

FAIL if it does not mention the vector, or raises only generic concerns such
as sparse IDs, unbounded growth, or thread safety without saying that Celeritas
assigns IDs in this range.
