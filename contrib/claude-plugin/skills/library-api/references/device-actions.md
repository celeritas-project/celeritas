# Custom device step actions (experimental)

A user step action runs a function on every active track slot at a chosen
point of each step, on CPU or GPU, with direct access to track state. The
worked example is `StepDiagnostic` in the offload template:
`example/offload-template/src/StepDiagnostic.hh`, `.cc`, `.cu`,
`StepDiagnosticData.hh`, `StepDiagnosticExecutor.hh`. The class itself
calls most of its code boilerplate that will be abstracted later, so expect
changes between releases.

## Structure

| Piece | Role |
|-------|------|
| `...Data.hh` | Params data (shared, immutable) and state data (per stream, mutable) as `Collection`s templated on ownership and memory space |
| `...Executor.hh` | A struct with a `CELER_FUNCTION void operator()(CoreTrackView const&)` that reads track state and updates the state data. This is the per-track kernel body. |
| Action class (`.hh`/`.cc`) | Derives from `CoreStepActionInterface` (step hook and `order()`, e.g. `StepActionOrder::user_post`), `AuxParamsInterface` (per-stream auxiliary state), and `ParamsDataInterface<...>` (host/device params). Provides a `make_and_insert(CoreParams const&, ...)` factory that registers the action and its aux data. |
| Host `step()` in `.cc` | `make_active_track_executor(...)` then `launch_action(*this, params, state, execute)` |
| Device `step()` in `.cu` | Same executor, launched with `ActionLauncher<decltype(execute)>` |

Register it at setup through the `add_user_actions` callback, which receives
`CoreParams const&`:
- Geant4 offload: the `SetupOptions` member of that name (as in
  `example/offload-template/src/MakeCelerOptions.cc`).
- Standalone: `inp::Diagnostics::add_user_actions`.

## Device-code rules

- Mark every function callable on the device `CELER_FUNCTION` (or
  `CELER_FORCEINLINE_FUNCTION`). No virtual calls, exceptions, `new`/`delete`,
  `std::string`, `std::vector`, or other host-only STL in device code.
- Pass data through `Collection`s and lightweight views captured by value in
  the executor, never through pointers to host objects.
- Accumulate with atomics (`corecel/math/Atomics.hh`) when several track
  slots update the same bin.
- Put kernel launches only in `.cu` files. The `.cc` file has the host path
  and is the only one compiled in CPU builds.
- Copy results back to the host through the state accessors the example
  provides. Do not dereference device memory on the host.

## Check

Build both CPU and GPU configurations when possible. Run with
`CELER_LOG=debug` and compare the accumulated results with an equivalent
Geant4 stepping action on a small problem.
