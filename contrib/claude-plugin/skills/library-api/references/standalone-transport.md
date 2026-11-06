# Programmatic standalone transport

This is what `celer-sim` does, driven from user C++ instead of a JSON file.
The reference implementation is `app/celer-sim/Runner.cc` (setup) and
`app/celer-sim/Transporter.cc` (stepping loop). Copy from the release tag
matching the installed Celeritas: the stepping API is evolving.

## 1. Describe the problem

Fill an `inp::StandaloneInput` (`celeritas/inp/StandaloneInput.hh`). Its
members are the same as the `celer-sim` JSON (`system`, `problem`,
`geant_setup`, `physics_import`, `events`). It can be built in code or read
from JSON with the functions in `celeritas/inp/StandaloneInputIO.json.hh`:

```cpp
celeritas::inp::StandaloneInput si;
nlohmann::json::parse(std::ifstream{"input.json"}).get_to(si);
```

Start from `celer-sim default` output to see every field (`standalone-input`
skill).

## 2. Build the problem

```cpp
auto loaded = celeritas::setup::standalone_input(si);
// loaded.problem.core_params : std::shared_ptr<CoreParams>
// loaded.problem.actions     : action sequence for the stepping loop
// loaded.events              : vector of events, each a vector<Primary>
```

`setup::standalone_input` (in `celeritas/setup/StandaloneInput.hh`) performs
the documented initialization order: system and GPU first, then geometry,
Geant4 physics import (when requested), and events.

## 3. Step

Each stream owns one `Stepper<MemSpace::host>` or
`Stepper<MemSpace::device>` (`celeritas/global/Stepper.hh`), built from a
`StepperInput`:

| `StepperInput` member | Value |
|-----------------------|-------|
| `params` | `loaded.problem.core_params` |
| `actions` | `loaded.problem.actions` |
| `stream_id` | `StreamId{i}` for stream `i` (< `params->sizes().streams`) |
| `primary_capacity` | At least the largest event's primary count |

Then transport each event the way `Transporter<M>::operator()` does in
`app/celer-sim/Transporter.cc`:

1. Push or stage the event's primaries and take the first step.
2. Keep stepping while the returned `StepperResult` reports active or
   queued tracks. Stop after a step limit, as a guard against stuck tracks.
3. Call `end_transport()` once the loop is empty, to complete deferred work
   such as buffered optical photons.

The `Stepper` exposes both an asynchronous interface (`async`, `ready`,
`wait`, `get`, `push_primary`, `stage_primaries`) and older synchronous call
operators that are marked deprecated. Use whichever the user's version of
`celer-sim` uses.

## 4. Output and diagnostics

- `CoreParams` holds an output registry. `celer-sim` writes it as the JSON
  diagnostic file (`app/celer-sim/RunnerOutput.cc` shows how), which the
  `troubleshoot` summarizer reads.
- To gather physics results, add a `StepCollector` with a `StepInterface`
  implementation before stepping, or use built-in user actions
  (`src/celeritas/user/`).
