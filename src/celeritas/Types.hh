//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.hh
//! Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/ThreadId.hh"
#include "orange/Types.hh"

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

//! Opaque index of electron subshell
using SubshellId = OpaqueId<struct Subshell>;

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
/*!
 * Physical state of matter.
 */
enum class MatterState
{
    unspecified = 0,
    solid,
    liquid,
    gas
};

//---------------------------------------------------------------------------//
//! Whether a track slot is alive, inactive, or dying
enum class TrackStatus : signed char
{
    killed   = -1, //!< Killed inside the step, awaiting replacement
    inactive = 0,  //!< No tracking in this thread slot
    alive    = 1   //!< Track is active and alive
};

//---------------------------------------------------------------------------//
// HELPER STRUCTS
//---------------------------------------------------------------------------//
//! Step length and limiting action to take
struct StepLimit
{
    real_type step{};
    ActionId  action{};

    //! Whether a step limit has been determined
    explicit CELER_FUNCTION operator bool() const
    {
        CELER_ASSERT(step >= 0);
        return static_cast<bool>(action);
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
