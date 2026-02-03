# Celeritas Optical Simulation System

This document provides an overview of the optical photon simulation architecture in Celeritas for AI agents and developers working on this codebase.

## Overview

The optical simulation system (`src/celeritas/optical`) handles propagation and interactions of optical photons in detectors. It follows the same data-oriented, GPU-compatible design patterns as the main Celeritas stepping loop but is specialized for optical physics.

## Core Architecture

### Track State Management

The optical system uses a simplified state structure compared to main particle tracking:

- **`CoreTrackView`**: Aggregates all track state (geometry, particle, simulation)
- **`SimTrackView`**: Simulation state (time, step length, status, step counters, primary ID)
- **`ParticleTrackView`**: Particle state (energy, polarization for optical photons)
- **`GeoTrackView`**: Geometry navigation state

### Data Structures

Following Celeritas conventions:

- **`SimParamsData<W, M>`**: Shared immutable parameters (max steps, iteration limits)
- **`SimStateData<W, M>`**: Mutable per-track state
  - `primary_ids`: Originating Geant4 primary (for offload correlation)
  - `time`: Lab-frame time elapsed since event start
  - `step_length`: Current step limit
  - `status`: Track status (inactive, initializing, alive, errored)
  - `post_step_action`: Action to execute at step end
  - `num_steps`: Total step count
- **Templates**: `W` = Ownership (value/reference/const_reference), `M` = MemSpace (host/device)

### Track Initialization

Tracks are initialized via `TrackInitializer`:
```cpp
struct TrackInitializer {
    units::MevEnergy energy;
    Real3 position, direction, polarization;
    real_type time;
    PrimaryId primary;
    ImplVolumeId volume;
};
```

## Photon Generation Sources

Optical photons originate from multiple sources:

### 1. Cherenkov Radiation (`CherenkovGenerator`)
- Generated when charged particles are above light speed in medium
- Uses material refractive index to compute emission spectrum
- Includes parent track's primary ID
- **Path**: `gen/CherenkovGenerator.{hh,cc}`

### 2. Scintillation (`ScintillationGenerator`)
- Produced from energy deposition in scintillating materials
- Samples from material-dependent time profiles and spectra
- Includes parent track's primary ID
- **Path**: `gen/ScintillationGenerator.{hh,cc}`

### 3. Wavelength Shifting (`WavelengthShiftGenerator`)
- Secondary photons from absorbed optical photons
- Inherits primary ID from parent optical track via `SimTrackView`
- **Path**: `interactor/WavelengthShiftGenerator.hh`

### 4. Primary Generation (`PrimaryGenerator`)
- User-configurable distributions (energy, angle, shape)
- For standalone testing without Geant4
- Can specify primary ID explicitly (defaults to invalid)
- **Path**: `gen/PrimaryGenerator.{hh,cc}`, `gen/PrimaryGeneratorAction.{hh,cc}`

### 5. Direct Generation (`DirectGenerator`)
- Direct initialization from pre-built `TrackInitializer` buffers
- Used for Geant4 offload: buffers populated by offload actions, consumed by direct generator
- Initializers stored in `DirectGeneratorStateData` and processed from back to front
- Includes primary ID from originating Geant4 track
- **Path**: `gen/DirectGeneratorData.hh`, `gen/detail/DirectGeneratorExecutor.hh`

## Primary ID Tracking

Primary IDs enable correlation between optical photons and their originating Geant4 primaries:

### Data Flow
1. **Offload**: Geant4 track → `GeneratorDistributionData.primary`
2. **Generation**: Cherenkov/Scintillation → `TrackInitializer.primary`
3. **Initialization**: `CoreTrackView::operator=(TrackInitializer)` → `SimTrackView::Initializer{primary, time}`
4. **Storage**: `SimStateData.primary_ids[track_slot]`
5. **Access**: `SimTrackView::primary_id()`
6. **Propagation**: WLS secondaries inherit from parent via `SimTrackView`

### Key Files
- [TrackInitializer.hh](TrackInitializer.hh): Track initialization data
- [SimData.hh](SimData.hh): Simulation state storage
- [SimTrackView.hh](SimTrackView.hh): Per-track simulation interface
- [GeneratorData.hh](gen/GeneratorData.hh): Distribution data for offload
- [WavelengthShiftData.hh](WavelengthShiftData.hh): WLS distribution data

## Action/Executor Pattern

Optical stepping uses the same three-layer pattern as main Celeritas:

1. **Action** (`StepActionInterface`): Manages execution order, launches kernels
2. **Executor**: Wraps interactor, handles track-level logic
3. **Interactor**: Pure physics functor operating on minimal state

Example: `AbsorptionModel` → `AbsorptionExecutor` → absorption physics

## Key Optical Processes

### Surface Physics
- Boundary interactions at optical surfaces
- Models: Dielectric reflection/refraction, roughness (polished, Gaussian, smear)
- **Path**: `surface/BoundaryAction.cc`, `surface/model/`

### Bulk Processes
- **Absorption**: Photon killed based on attenuation length
- **Rayleigh scattering**: Elastic scattering in bulk material
- **Mie scattering**: Scattering from particle suspensions
- **Wavelength shifting**: Absorption followed by re-emission
- **Path**: `model/` directory

## Generator Registry

`GeneratorRegistry` manages multiple photon sources:
- Assigns unique `GeneratorId` to each source
- Tracks buffer space requirements
- Coordinates generation across actions

## Important Differences from Main Stepping

1. **No particle types**: Only optical photons (single particle type)
2. **Simplified state**: No secondary storage, simplified initialization
3. **Direct generation**: Photons created directly, not from secondaries stack
4. **Surface-dominated**: Optical surfaces are primary interaction mode

## Geometry Integration

Optical materials (`OpticalMaterialParams`) map to geometry volumes.

## GPU Execution

All optical code is host-device compatible (`CELER_FUNCTION` macros):
- Launch via `launch_action()` or device-specific launchers
- Collections manage host/device data transfer
- Explicit memory space handling (MemSpace::host/device)

## File Organization

```
src/celeritas/optical/
├── Core*.{hh,cc}           # Core parameters, state, track view
├── SimData.hh              # Simulation state structures
├── SimParams.{hh,cc}       # Simulation parameters
├── SimTrackView.hh         # Per-track simulation interface
├── TrackInitializer.hh     # Track initialization data
├── action/                 # Step actions (along-step, pre-step, discrete)
├── gen/                    # Photon generators (Cherenkov, scintillation, primary)
│   ├── *Generator.{hh,cc}
│   ├── *Action.{hh,cc}
│   └── detail/             # Implementation details, executors
├── interactor/             # Process interactors
├── model/                  # Physics models (absorption, scattering, WLS)
└── surface/                # Surface physics (boundary, models)
```

## Key Conventions

1. **Initialization**: All `TrackInitializer` instances must set all fields (including `primary`)
2. **Validity**: Use `explicit operator bool()` for validation
3. **Assignment**: Support cross-memory-space assignment via templated `operator=`
4. **Collections**: Use `ItemId<T>`, `ItemRange<T>` for type-safe indexing
5. **Units**: Energy is `units::MevEnergy`, positions are `Real3` in cm by default

## Related Documentation

- Main AGENTS.md (repository root): General Celeritas patterns
- Action/Executor/Interactor paradigm: See main stepping loop examples
- Collections guide: `src/corecel/data/Collection.hh`
- Testing: `test/celeritas/optical/` directory

## Common Pitfalls

1. **Missing CELER_FUNCTION**: Host-device code requires proper macros
2. **Data/execution inconsistency**: Check that input, data, views, initializers, and executors match
3. **Aggregate initialization**: Field order matters! Check structure definitions.

## Future Work / TODO

- Correlate optical photon detection with originating primaries
- Enhanced optical photon scoring and output
- Multi-event buffer management optimization
