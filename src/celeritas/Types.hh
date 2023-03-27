//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.hh
//! \brief Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

// IWYU pragma: begin_exports
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/ThreadId.hh"
#include "orange/Types.hh"
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
using EventId = OpaqueId<struct Event>;

//! Opaque index to MaterialRecord in a vector: represents a material ID
using MaterialId = OpaqueId<struct MaterialRecord>;

//! Opaque index of model in the list of physics processes
using ModelId = OpaqueId<class Model>;

//! Opaque index to ParticleRecord in a vector: represents a particle type
using ParticleId = OpaqueId<struct ParticleRecord>;

//! Opaque index of physics process
using ProcessId = OpaqueId<class Process>;

//! Unique ID (for an event) of a track among all primaries and secondaries
using TrackId = OpaqueId<struct Track>;

//---------------------------------------------------------------------------//
// (detailed type aliases)
//---------------------------------------------------------------------------//

//! Opaque index to one elemental component datum in a particular material
using ElementComponentId = OpaqueId<struct MatElementComponent>;

//! Opaque index of a process applicable to a single particle type
using ParticleProcessId = OpaqueId<ProcessId>;

//! Opaque index of a model applicable to a single particle type
using ParticleModelId = OpaqueId<ModelId>;

//! Opaque index of electron subshell
using SubshellId = OpaqueId<struct Subshell>;

//! Opaque index for mapping volume-specific "sensitive detector" objects
using DetectorId = OpaqueId<struct Detector>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
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
    gas
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
    pre,  //!< Pre-step physics and setup
    along,  //!< Along-step
    pre_post,  //!< Discrete selection kernel
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
    unsorted,
    shuffled,
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

//! Placeholder for helper functions whose interface requires data
struct NoData
{
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to a surface type
char const* to_cstring(ActionOrder);

//---------------------------------------------------------------------------//
}  // namespace celeritas
