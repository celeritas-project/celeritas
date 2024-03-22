//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/grid/GenericGridData.hh"

/* Notes:
 *
 * Every optical physics process is defined over a single energy range which is
 * just the optical photon energy range. Therefore there are just optical
 * processes with no models (Celeritas models seperatae physics of a given
 * process over multiple energy ranges. Equivalently there's a unique model
 * for every process, so we may just identify that unique model with the
 * process itself).
 *
 * Only macro_xs grid types (no energy loss or range).
 */
namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
//! Currently all value grids are cross section grids
using OpticalValueGrid = GenericGridData;
using OpticalValueGridId = OpaqueId<GenericGridData>;
using OpticalValueTableId = OpaqueId<struct OpticalValueTable>;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Set of value grids for all elements or materials.
 *
 * It is allowable for this to be "false" (i.e. no materials assigned)
 * indicating that the value table doesn't apply in the context -- for
 * example, an empty ValueTable macro_xs means that the process doesn't have a
 * discrete interaction.
 */
struct OpticalValueTable
{
    ItemRange<OpticalValueGridId> grids;  //!< Value grid by element or material index

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !grids.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Processes for optical photons.
 */
struct OpticalProcessGroup
{
    ItemRange<ProcessId> processes;  //!< Processes that apply [ppid]
    ItemRange<OpticalValueTable> lambda_tables;  //!< [ppid]

    // photons are never at rest
    // bool has_at_rest{};

    // no energy loss process
    // ParticleProcessId eloss_ppid{};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !processes.empty();
    }

    //! Number of processes that apply
    CELER_FUNCTION ProcessId::size_type size() const
    {
        return processes.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar (no template needed) quantities used by physics.
 *
 * The user-configurable constants are described in \c PhysicsParams .
 *
 * The \c process_to_action value corresponds to the \c ActionId for the first \c
 * ProcessId . Additionally it implies (by construction in physics_params) the
 * action IDs of several other physics actions.
 */
struct OpticalPhysicsParamsScalars
{
    using Energy = units::MevEnergy;

    //! Highest number of processes for any particle type
    ProcessId::size_type max_particle_processes{};
    //! Offset to create an ActionId from a ProcessId
    ActionId::size_type process_to_action{};
    //! Number of physics models
    ProcessId::size_type num_processes{};

    // User-configurable constants
    real_type secondary_stack_factor = 1;  //!< Secondary storage per state
                                           //!< size

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_particle_processes > 0
               && process_to_action >= 1
               && num_processes > 0
               && secondary_stack_factor > 0;
    }

    CELER_FORCEINLINE_FUNCITON ActionId discrete_action() const
    {
        return ActionId{process_to_action - 1};
    }

    //! Indicate an interaction failed to allocate memory
    CELER_FORCEINLINE_FUNCTION ActionId failure_action() const
    {
        return ActionId{process_to_action + num_processes};
    }
};

//---------------------------------------------------------------------------//
/*!
 */
template<Ownership W, MemSpace M>
struct OpticalPhysicsParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ProcessItems = Collection<T, W, M, ProcessId>;

    //// DATA ////

    // Backend storage
    Items<real_type> reals;
    Items<OpticalValueGrid> value_grids;
    Items<OpticalValueGridId> value_grid_ids;
    Items<OpticalValueTable> value_tables;
    Items<OpticalValueTableId> value_table_ids;
    Items<ProcessId> process_ids;
    OpticalProcessGroup process_groups;

    // Non-templated data
    OpticalPhysicsParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !process_ids.empty() && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalPhysicsParamsData& operator=(OpticalPhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        reals = other.reals;
        value_grids = other.value_grids;
        value_grid_ids = other.value_grid_ids;
        process_ids = other.process_ids;
        value_tables = other.value_tables;
        value_table_ids = other.value_table_ids;
        process_groups = other.process_groups;

        scalars = other.scalars;

        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 */
struct OpticalPhysicsTrackState
{
    real_type interaction_mfp;  //!< Remaining MFP to interaction

    // TEMPORARY STATE
    real_type macro_xs;  //!< Total cross section for discrete interactions
    real_type energy_deposition;  //!< Local energy deposition in a step [MeV]
    Span<Secondary> secondaries;  //!< Emitted secondaries
    ElementComponentId element;  //!< Element sampled for interaction
};

//---------------------------------------------------------------------------//
/*!
 */
struct OpticalPhysicsTrackInitializer
{
};

//---------------------------------------------------------------------------//
/*!
 */
template<Ownership W, MemSpace M>
struct OpticalPhysicsStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    StateItems<OpticalPhysicsTrackState> state;  //!< Track state [track]

    Items<real_type> per_process_xs;  //!< XS [track][particle process]

    StackAllocatorData<Secondary, W, M> secondaries;  //!< Secondary stack

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !state.empty() && secondaries;
    }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    OpticalPhysicsStateData& operator=(OpticalPhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;

        per_process_xs = other.per_process_xs;

        secondaries = other.secondaries;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(OpticalPhysicsStateData<Ownership::value, M>* state,
                   HostCRef<OpticalPhysicsParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&state->state, size);
    resize(&state->per_process_xs,
           size * params.scalars.max_particle_processes);
    resize(
        &state->secondaries,
        static_cast<size_type>(size * params.scalars.secondary_stack_factor));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
