# Sensitive detectors and hits

By default Celeritas finds every logical volume with a Geant4 sensitive
detector (SD). For each step that deposits energy in one, it rebuilds a
`G4Step` and calls the SD's `ProcessHits` on the worker thread that
offloaded the event. Existing SD and hit code therefore works unchanged, but
only with the data Celeritas is asked to provide.

The authority is `SDSetupOptions` in the installed `accel/SetupOptions.hh`.
The option is named `opts.sd`.

## Decide what the SDs read

Read each `ProcessHits` implementation in the user's code and list what it
uses: energy deposit, step length, pre/post position, time, direction,
kinetic energy, `GetTrack()`, the touchable (copy numbers, replica numbers,
`GetVolume(depth)`), material.

Then disable everything else. Each enabled item costs memory and transfer
time:

| Option | Provides | Disable when |
|--------|----------|--------------|
| `opts.sd.enabled` | Any SD callbacks | The geometry has no SDs. Setup fails with "no G4 sensitive detectors are defined" otherwise. |
| `opts.sd.ignore_zero_deposition` | Skip steps with zero local energy deposit (default on) | Turn off only if SDs count zero-deposit steps (e.g. fluence or crossing counters). |
| `opts.sd.energy_deposition` | `G4Step::GetTotalEnergyDeposit` | Never needed by the SD (rare) |
| `opts.sd.step_length` | `G4Step::GetStepLength` | Unused |
| `opts.sd.locate_touchable` | Pre-step `G4TouchableHandle` (copy numbers, replicas) | SDs only use the logical volume. This is the **most expensive** option, so turn it off whenever possible. |
| `opts.sd.locate_touchable_post` | Post-step touchable | Unused |
| `opts.sd.track` | A `G4Track` with particle type and post-step data | SDs never call `GetTrack()` (then it returns `nullptr`) |
| `opts.sd.pre` / `opts.sd.post` | Per-step-point `global_time`, `position`, `direction`, `kinetic_energy` | Disable individual fields that are unused |

`opts.sd.force_volumes` and `opts.sd.skip_volumes` take sets of
`G4LogicalVolume const*`. Build them with `celeritas::FindVolumes({"name1",
"name2"})`:

- `force_volumes`: SDs exist only on worker threads (unusual run managers).
- `skip_volumes`: volumes whose scoring is done on the GPU by a custom
  Celeritas action instead of a Geant4 callback.

## MC truth caveats (tell the user)

- Hits from tracks Geant4 offloaded keep the original track ID, parent ID,
  creator process, and `G4VUserTrackInformation`.
- Tracks created *inside* Celeritas (secondaries) get IDs counting down from
  `INT_MAX`, the ID of their real parent, a placeholder creator process named
  `celeritas`, and **no** user track information.
- Geant4 never sees Celeritas-created tracks: there are no user tracking or
  stepping action calls and no trajectories. Truth records that depend on
  `UserTrackingAction`, `UserSteppingAction`, or trajectory storage will be
  incomplete for offloaded showers. Plan for this, for example by recording
  truth only at the offload boundary.
- `AuxiliaryTrackInformation` is never set.
- Only `ProcessHits` is called back. Other Geant4 user hooks are not invoked
  for steps inside Celeritas.

## Verify

Compare per-detector summed energy deposition and hit counts between
`CELER_DISABLE=1` and Celeritas runs (see `validation.md`). A detector with
zero Celeritas hits usually means a missing SD on the master thread (use
`force_volumes`) or a filter in `ProcessHits` that depends on data that was
disabled.
