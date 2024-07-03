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

//! Opaque index of a material modified by physics options
// TODO: rename to PhysMatId; equivalent to "material cuts couple"
using MaterialId = OpaqueId<class Material_>;

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

//! Opaque index of a process applicable to a single particle type
using ParticleProcessId = OpaqueId<ProcessId>;

//! Opaque index of a model applicable to a single particle type
using ParticleModelId = OpaqueId<ModelId>;

//! Opaque index of electron subshell
using SubshellId = OpaqueId<struct Subshell_>;

//! Opaque index of particle-nucleon cascade channel
using ChannelId = OpaqueId<struct Channel_>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Interpolation type
enum class Interp
{
    linear,
    log,
    size_
};

//---------------------------------------------------------------------------//
//! Physical state of matter
enum class MatterState
{
    unspecified = 0,
    solid,
    liquid,
    gas,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Whether a track slot is alive, inactive, or dying inside a step iteration.
 *
 * - A track slot starts as \c inactive . If not filled with a new track, it is
 *   inactive for the rest of the step iteration.
 * - When it is populated with a new particle, it is \c initializing . If an
 *   error occurs during initialization it is \c errored .
 * - During the pre-step setup, a non-errored active track slot is marked as \c
 *   alive .
 * - During along-step or post-step a track can be marked as \c errored or
 *   \c killed .
 */
enum class TrackStatus : std::uint_least8_t
{
    inactive = 0,  //!< No tracking in this thread slot
    initializing,  //!< Before pre-step, after initialization
    alive,  //!< Track is active and alive
    begin_dying_,
    errored = begin_dying_,  //!< Track failed during this step
    killed,  //!< Killed physically inside the step
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Within-step ordering of explicit actions.
 *
 * Each "step iteration", wherein many tracks undergo a single step in
 * parallel, consists of an ordered series of actions. An action with an
 * earlier order always precedes an action with a later order.
 *
 * \sa ExplicitActionInterface
 */
enum class ActionOrder
{
    start,  //!< Initialize tracks
    user_start,  //!< User initialization of new tracks
    sort_start,  //!< Sort track slots after initialization
    pre,  //!< Pre-step physics and setup
    user_pre,  //!< User actions for querying pre-step data
    sort_pre,  //!< Sort track slots after setting pre-step
    along,  //!< Along-step
    sort_along,  //!< Sort track slots after determining first step action
    pre_post,  //!< Discrete selection kernel
    sort_pre_post,  //! Sort track slots after selecting discrete interaction
    post,  //!< After step
    user_post,  //!< User actions after boundary crossing, collision
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
    shuffled,  //!< Shuffle at the start of the simulation

    partition_status,  //!< Partition by status at the start of each step
    sort_along_step_action,  //!< Sort only by the along-step action id
    sort_step_limit_action,  //!< Sort only by the step limit action id
    sort_action,  //!< Sort by along-step id, then post-step ID
    sort_particle_type,  //!< Sort by particle type
    size_
};

//---------------------------------------------------------------------------//
//! Algorithm used to calculate the multiple scattering step limit
enum class MscStepLimitAlgorithm
{
    minimal,
    safety,
    safety_plus,
    distance_to_boundary,
    size_,
};

//---------------------------------------------------------------------------//
//! Nuclear form factor model for Coulomb scattering
enum class NuclearFormFactorType
{
    none,
    flat,
    exponential,
    gaussian,
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
//! Action order/ID tuple for comparison in sorting
struct OrderedAction
{
    ActionOrder order;
    ActionId id;

    //! Ordering comparison for an action/ID
    CELER_CONSTEXPR_FUNCTION bool operator<(OrderedAction const& other) const
    {
        if (this->order < other.order)
            return true;
        if (this->order > other.order)
            return false;
        return this->id < other.id;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//! Whether a track is in a consistent, valid state
CELER_CONSTEXPR_FUNCTION bool is_track_valid(TrackStatus status)
{
    return status != TrackStatus::inactive && status != TrackStatus::errored;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//

// Get a string corresponding to an interpolation
char const* to_cstring(Interp);

// Get a string corresponding to a material state
char const* to_cstring(MatterState);

// Get a string corresponding to a track stats
char const* to_cstring(TrackStatus);

// Get a string corresponding to a surface type
char const* to_cstring(ActionOrder);

// Get a string corresponding to a track ordering policy
char const* to_cstring(TrackOrder);

// Get a string corresponding to the MSC step limit algorithm
char const* to_cstring(MscStepLimitAlgorithm value);

// Get a string corresponding to the nuclear form factor model
char const* to_cstring(NuclearFormFactorType value);

// Whether the TrackOrder will sort tracks by actions with the given
// ActionOrder
bool is_action_sorted(ActionOrder action, TrackOrder track);

//! Whether track sorting is enabled
inline constexpr bool is_action_sorted(TrackOrder track)
{
    return static_cast<int>(track) > static_cast<int>(TrackOrder::shuffled);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
