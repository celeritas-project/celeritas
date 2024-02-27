//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.hh
//! \brief Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "celeritas_config.h"
// IWYU pragma: begin_exports
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"
// IWYU pragma: end_exports

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! End-of-step (or perhaps someday within-step?) action to take
using ActionId = OpaqueId<class ActionInterface>;

//! Opaque index to ElementRecord in the global vector of elements
using ElementId = OpaqueId<struct ElementRecord>;

//! Counter for the initiating event for a track
using EventId = OpaqueId<struct Event_>;

//! Opaque index to IsotopeRecord in a vector
using IsotopeId = OpaqueId<struct IsotopeRecord>;

//! Opaque index of model in the list of physics processes
using ModelId = OpaqueId<class Model>;

//! Opaque index to ParticleRecord in a vector: represents a particle type
using ParticleId = OpaqueId<struct Particle_>;

//! Opaque index of physics process
using ProcessId = OpaqueId<class Process>;

//! Unique ID (for an event) of a track among all primaries and secondaries
using TrackId = OpaqueId<struct Track_>;

//---------------------------------------------------------------------------//
// (detailed type aliases)
//---------------------------------------------------------------------------//

//! Opaque index for mapping volume-specific "sensitive detector" objects
using DetectorId = OpaqueId<struct Detector_>;

//! Opaque index to one elemental component datum in a particular material
using ElementComponentId = OpaqueId<struct MatElementComponent>;

//! Opaque index to one isotopic component datum in a particular element
using IsotopeComponentId = OpaqueId<struct ElIsotopeComponent>;

//! Opaque index to a material with optical properties
using OpticalMaterialId = OpaqueId<struct OpticalMaterial_>;

//! Opaque index of a process applicable to a single particle type
using ParticleProcessId = OpaqueId<ProcessId>;

//! Opaque index of a model applicable to a single particle type
using ParticleModelId = OpaqueId<ModelId>;

//! Opaque index of electron subshell
using SubshellId = OpaqueId<struct Subshell_>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Unit system used by Celeritas
enum class UnitSystem
{
    none,  //!< Invalid unit system
    cgs,  //!< Gaussian CGS
    si,  //!< International System
    clhep,  //!< Geant4 native
    size_,
    native = CELERITAS_UNITS,  //!< Compile time selected system
};

//---------------------------------------------------------------------------//
//! Interpolation type
enum class Interp
{
    linear,
    log
};

//---------------------------------------------------------------------------//
//! Physical state of matter.
enum class MatterState
{
    unspecified = 0,
    solid,
    liquid,
    gas,
    size_
};

//---------------------------------------------------------------------------//
//! Whether a track slot is alive, inactive, or dying
enum class TrackStatus : std::int_least8_t
{
    killed = -1,  //!< Killed inside the step, awaiting replacement
    inactive = 0,  //!< No tracking in this thread slot
    // TODO: add 'initial' enum here, change "alive" to helper function
    alive = 1  //!< Track is active and alive
};

//---------------------------------------------------------------------------//
//! Within-step ordering of explicit actions
enum class ActionOrder
{
    start,  //!< Initialize tracks
    sort_start,  //!< Sort track slots after initialization
    pre,  //!< Pre-step physics and setup
    sort_pre,  //!< Sort track slots after setting pre-step
    along,  //!< Along-step
    sort_along,  //!< Sort track slots after determining first step action
    pre_post,  //!< Discrete selection kernel
    sort_pre_post,  //! Sort track slots after selecting discrete interaction
    post,  //!< After step
    post_post,  //!< User actions after boundary crossing, collision
    end,  //!< Processing secondaries, including replacing primaries
    size_
};

//---------------------------------------------------------------------------//
//! Differentiate between result data at the beginning and end of a step.
enum class StepPoint
{
    pre,
    post,
    size_
};

//---------------------------------------------------------------------------//
//! Ordering / sorting of tracks on GPU
enum class TrackOrder
{
    unsorted,  //!< Don't do any sorting: tracks are in an arbitrary order
    shuffled,  //!< Tracks are shuffled at the start of the simulation
    partition_status,  //!< Tracks are partitioned by status at the start of
                       //!< each step
    sort_along_step_action,  //!< Sort only by the along-step action id
    sort_step_limit_action,  //!< Sort only by the step limit action id
    sort_action,  //!< Sort by along-step id, then post-step ID
    sort_particle_type,  //!< Sort by particle type
    size_
};

//---------------------------------------------------------------------------//
// HELPER STRUCTS
//---------------------------------------------------------------------------//
//! Step length and limiting action to take
struct StepLimit
{
    real_type step{};
    ActionId action{};

    //! Whether a step limit has been determined
    explicit CELER_FUNCTION operator bool() const
    {
        CELER_ASSERT(step >= 0);
        return static_cast<bool>(action);
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to a unit system
char const* to_cstring(UnitSystem);

// Get a unit system corresponding to a string
UnitSystem to_unit_system(std::string const& s);

// Get a string corresponding to a material state
char const* to_cstring(MatterState);

// Get a string corresponding to a surface type
char const* to_cstring(ActionOrder);

// Get a string corresponding to a track ordering policy
char const* to_cstring(TrackOrder);

// Checks that the TrackOrder will sort tracks by actions applied at the given
// ActionOrder
bool is_action_sorted(ActionOrder action, TrackOrder track);

//---------------------------------------------------------------------------//
}  // namespace celeritas
