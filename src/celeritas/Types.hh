//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/Types.hh
//! \brief Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Config.hh"
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

//! Opaque index to ElementRecord in the global vector of elements
using ElementId = OpaqueId<struct ElementRecord>;

//! Zero-indexed counter for the initiating event for a track
using EventId = OpaqueId<struct Event_>;

//! Unique identifier for an event used by external applications
using UniqueEventId = OpaqueId<struct Event_, std::uint64_t>;

//! Opaque index to IsotopeRecord in a vector
using IsotopeId = OpaqueId<struct IsotopeRecord>;

//! Opaque index of a material modified by physics options
using PhysMatId = OpaqueId<struct PhysicsMaterial_>;

//! Opaque index of model in the list of physics processes
using ModelId = OpaqueId<struct Model_>;

//! Opaque index to a material with optical properties
using OptMatId = OpaqueId<struct OpticalMaterial_>;

//! Opaque index to ParticleRecord in a vector: represents a particle type
using ParticleId = OpaqueId<struct Particle_>;

//! Unique ID (for an event) of a track among all primaries
using PrimaryId = OpaqueId<struct Primary_>;

//! Opaque index of physics process
using ProcessId = OpaqueId<struct Process_>;

//! Unique ID (for an event) of a track among all primaries and secondaries
using TrackId = OpaqueId<struct Track_>;

//---------------------------------------------------------------------------//
// (detailed type aliases)
//---------------------------------------------------------------------------//

//! Opaque index of particle-nucleon cascade channel
using ChannelId = OpaqueId<struct Channel_>;

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

//! Opaque index of a uniform grid
using UniformGridId = OpaqueId<struct UniformGridRecord>;

//! Opaque index of a cross section grid
using XsGridId = OpaqueId<struct XsGridRecord>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
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
 * Each track slot has a state marking its transition between death and life.
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
//! Differentiate between result data at the beginning and end of a step.
enum class StepPoint
{
    pre,
    post,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Change the ordering or thread mapping of track slots.
 *
 * There are three categories of track sorting:
 * 1. No sorting is performed and new tracks are inserted in the nearest empty
 *    track slot. (\c none )
 * 2. The location of new track slots is biased during track initialization:
 *    charged and neutral tracks are inserted at opposite sides of the track
 *    slot vacancies. (\c init_charge )
 * 3. Tracks are \em reindexed one or more times per step so that the layout
 *    in memory is unchanged but an additional indirection maps threads onto
 *    different track slots based on particle attributes (\c reindex_status,
 *    \c reindex_particle_type ), actions (\c reindex_along_step_action,
 *    \c reindex_step_limit_action, \c reindex_both_action ).
 * 4. As a control to measure the cost of indirection, the track slots can be
 *    reindexed randomly at the beginning of execution (\c reindex_shuffle ).
 */
enum class TrackOrder
{
    none,  //!< Don't do any sorting: tracks are in an arbitrary order
    begin_layout_,
    //! Partition data layout of new tracks by charged vs neutral
    init_charge = begin_layout_,
    end_layout_,
    begin_reindex_ = end_layout_,
    //!< Shuffle at the start of the simulation
    reindex_shuffle = begin_reindex_,
    reindex_status,  //!< Partition by active/inactive status
    reindex_particle_type,  //!< Sort by particle type
    begin_reindex_action_,
    //! Sort only by the along-step action id
    reindex_along_step_action = begin_reindex_action_,
    reindex_step_limit_action,  //!< Sort only by the step limit action id
    reindex_both_action,  //!< Sort by along-step id, then post-step ID
    end_reindex_action_,
    end_reindex_ = end_reindex_action_,
    size_ = end_reindex_
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
//! Optical photon wavelength shifting time model
enum class WlsTimeProfile
{
    delta,  //!< Delta function
    exponential,  //!< Exponential decay
    size_
};

//---------------------------------------------------------------------------//
//! Interpolation for physics grids
enum class InterpolationType
{
    linear,
    poly_spline,  //!< Piecewise polynomial interpolation
    cubic_spline,  //!< Cubic spline interpolation with \f$ C^2 \f$ continuity
    size_
};

//---------------------------------------------------------------------------//
//! Cylindrical coordinates indices
enum class CylAxis
{
    r = 0,
    phi,
    z,
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

// Get a string corresponding to a material state
char const* to_cstring(MatterState);

// Get a string corresponding to a track stats
char const* to_cstring(TrackStatus);

// Get a string corresponding to a track ordering policy
char const* to_cstring(TrackOrder);

// Get a string corresponding to the MSC step limit algorithm
char const* to_cstring(MscStepLimitAlgorithm value);

// Get a string corresponding to the nuclear form factor model
char const* to_cstring(NuclearFormFactorType value);

// Get a string corresponding to the interpolation method
char const* to_cstring(InterpolationType value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
