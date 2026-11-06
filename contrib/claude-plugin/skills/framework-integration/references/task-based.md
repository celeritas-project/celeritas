# Custom and task-based frameworks (Gaudi, CMSSW, Athena, ...)

Framework internals vary between versions. Locate the hooks in the user's
framework source instead of assuming class names, and confirm them with the
user. The Celeritas side is below; every name is declared in the installed
`accel/` headers.

## High-level or low-level?

| | High-level: `TrackingManagerIntegration` | Low-level: `SharedParams` + `LocalTransporter` |
|-|------------------------------------------|-----------------------------------------------|
| Objects | Process-wide singleton (`Instance()`) | Framework owns a shared `SharedParams` and one `LocalTransporter` per worker |
| Setup/teardown | Automatic through Geant4 state hooks | Framework calls them explicitly (see below) |
| UI macros, MPI init, logger setup | Included | Do it yourself if needed |
| Use when | The framework uses a standard Geant4 run manager (serial, MT, or tasking) | Singletons are forbidden, the framework drives transport itself, or offload must be controlled per event/stream |

Start with the high-level class unless there is a concrete reason not to.

### High-level outline

```cpp
// Where the physics list is assembled (once):
auto& tmi = celeritas::TrackingManagerIntegration::Instance();
physics_list->RegisterPhysics(new celeritas::TrackingManagerConstructor(&tmi));

// On the master thread, after the run manager exists, before initialization:
tmi.SetOptions(MakeCelerOptionsFromFrameworkConfig(config));
```

### Low-level outline (mirrors `app/celer-g4`)

```cpp
// Master, during setup: one shared object for all threads
auto shared = std::make_shared<celeritas::SharedParams>();
// Physics list: map Geant4 thread IDs to the thread-local transporter
physics_list->RegisterPhysics(new celeritas::TrackingManagerConstructor(
    shared.get(), [](int) { return &thread_local_transporter(); }));

// Master, at begin of run (after Geant4 physics tables exist):
shared->Initialize(options);
// Each worker, at begin of run:
shared->InitializeWorker(options);
thread_local_transporter().Initialize(options, *shared);
// Each worker, per event:
transporter.InitializeEvent(event_id);  // at event start
transporter.Flush();                    // at event end, before hits are read
// Each worker, end of run, on the SAME thread that created it:
transporter.Finalize();
// Master, end of run:
shared->Finalize();                     // writes diagnostics
```

Reference implementation: `app/celer-g4/RunAction.cc`,
`app/celer-g4/EventAction.cc`, `app/celer-g4/ActionInitialization.cc`.
`LocalTransporter` uses Geant4 thread-local allocators, so it **must** be
created, used, and finalized on one thread. Check
`SharedParams::GetMode()` before calling into the transporter: it is
`disabled` under `CELER_DISABLE`, and in kill-offload mode tracks are
killed instead of transported.

## Threads, tasks, and streams

- Celeritas allocates one stream (state) per Geant4 worker thread. The
  number comes from the run manager. Task-based run managers may report
  **zero** threads, and frameworks may create more worker threads than the
  run manager reports.
- Set `opts.get_num_streams` to return the true maximum number of
  concurrent Geant4 worker threads. Otherwise setup fails with "Geant4
  ThreadID (N) is out of range for the reported number of worker threads".
- `max_num_tracks` and `initializer_capacity` are per stream, so total memory
  scales with the stream count. With many CPU threads sharing one GPU,
  reduce per-stream capacities accordingly.
- Do not combine track-level OpenMP Celeritas builds with multithreaded
  Geant4 (Celeritas warns about this).

## Configuration plumbing

- Map framework parameters onto `SetupOptions` in one function: track slots,
  initializer capacity, ignored processes, field, SD options, output file.
  `src/ddceler/CelerPhysics.cc` (DDG4 properties) and
  `src/larceler/pdfullsimceler.fcl` (FHiCL) are worked examples.
- Frameworks often scrub or fix the environment. Instead of relying on
  `CELER_*` environment variables, insert values before Celeritas is set up
  for the first time, on one thread:
  `celeritas::environment().insert({"CELER_NONFATAL_FLUSH", "1"});`
  (declared in `corecel/sys/Environment.hh`). The variables consulted are
  recorded in the diagnostic output.

## Logging and errors

- Celeritas logs through two loggers: `celeritas::world_logger()` (setup
  messages, once per process) and `celeritas::self_logger()` (per-thread
  messages). The high-level integration replaces both with Geant4-aware
  loggers the first time `Instance()` runs. To route messages into the
  framework's message service, set a handler *after* that first call:

  ```cpp
  celeritas::world_logger().handle(
      [](celeritas::LogProvenance prov, celeritas::LogLevel lev, std::string msg) {
          // forward to the framework logger with prov.file / prov.line
      });
  ```

  Do the same for `self_logger()`. The handler must be thread-safe.
- Celeritas errors inside Geant4 callbacks become fatal `G4Exception`s with
  `celer.*` codes, so the framework's `G4VExceptionHandler` sees them. In
  low-level code, wrap calls the same way:
  `CELER_TRY_HANDLE(transporter.Flush(), celeritas::ExceptionConverter{"myfw.flush"});`.
- For production jobs that must not abort on stuck tracks, consider
  `CELER_NONFATAL_FLUSH`, which kills and reports the active tracks instead.

## GPUs and processes

- Each process uses one GPU. With MPI, Celeritas picks device
  `rank % num_devices`. Without MPI (or with `CELER_DISABLE_PARALLEL=1`)
  every process uses the first *visible* device. Several processes per node
  therefore need distinct `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`, or
  the batch system's GPU binding.
- Many processes sharing one GPU work, but each holds its own Celeritas state
  in device memory. Size capacities for the per-process share.

## MC truth

Celeritas-created secondaries never appear as Geant4 tracks. They trigger no
tracking/stepping actions and store no trajectories, and their track IDs
count down from `INT_MAX` (details in the `geant4-offload` skill's
`references/sensitive-detectors.md`). Frameworks that build truth trees from
tracking actions must either accept truth truncated at the offload boundary
or record what they need from SD hits.
