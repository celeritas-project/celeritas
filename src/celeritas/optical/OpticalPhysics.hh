//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysics.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

#include "OpticalPrimary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
//! Currently all value grids are cross section grids
using OpticalValueGrid = GenericGridData;
using OpticalValueGridId = OpaqueId<GenericGridData>;
using OpticalValueTableId = OpaqueId<struct OpticalValueTable>;
using OpticalSecondary = OpticalPrimary;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * IDs of OpticalValueGrids stored in OpticalPhysicsParamsData.
 *
 * The grids are indexed by material, and each optical process will have its
 * own table.
 */
struct OpticalValueTable
{
    ItemRange<OpticalValueGridId> grids;  //!< Value grid by material index

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !grids.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Per process tables for optical photons.
 *
 * Each process is assigned an optical value table that corresponds to the
 * macro cross section of the process (there is no energy loss, range, etc.)
 */
struct OpticalProcessGroup
{
    ItemRange<OpticalValueTable> macro_xs_tables;  //!< [opid]

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !macro_xs_tables.empty();
    }

    //! Number of processes that apply
    CELER_FUNCTION ProcessId::size_type size() const
    {
        return macro_xs_tables.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Named scalar (no template needed) quantities for optical physics.
 *
 * User-configurable constants are described \cd OpticalPhysicsParams .
 */
struct OpticalPhysicsParamsScalars
{
    using Energy = units::MevEnergy;

    //! Offset to create an ActionId from a OpticalProcessId
    ActionId::size_type process_to_action{};

    //! Number of optical processes
    OpticalProcessId::size_type num_processes{};

    // User-configurable constants
    real_type secondary_stack_factor = 1;  //!< Secondary storage per state
                                           //!< size

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return process_to_action >= 1 && num_processes > 0
               && secondary_stack_factor > 0;
    }

    //! Undergo a discrete interaction
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
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
 * Persistent shared optical physics data.
 */
template<Ownership W, MemSpace M>
struct OpticalPhysicsParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    // Backend storage
    Items<real_type> reals;
    Items<OpticalValueGrid> value_grids;
    Items<OpticalValueGridId> value_grid_ids;
    Items<OpticalValueTable> value_tables;
    Items<OpticalValueTableId> value_table_ids;
    Items<OpticalProcessId> process_ids;
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
 * Optical physics state data for a single track.
 */
struct OpticalPhysicsTrackState
{
    real_type interaction_mfp;  //!< Remaining MFP to interaction

    // TEMPORARY STATE
    real_type macro_xs;  //!< Total cross section for discrete interactions
    real_type energy_deposition;  //!< Local energy deposition in a step [MeV]
    Span<OpticalSecondary> secondaries;  //!< Emitted secondaries
    ElementComponentId element;  //!< Element sampled for interaction
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an optical physics state track.
 */
struct OpticalPhysicsTrackInitializer
{
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic optical physics (models, processes) state data.
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

    Items<real_type> per_process_xs;  //!< XS [track]

    StackAllocatorData<OpticalSecondary, W, M> secondaries;  //!< Secondary
                                                             //!< stack

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
    resize(&state->per_process_xs, size * params.scalars.num_processes);
    resize(
        &state->secondaries,
        static_cast<size_type>(size * params.scalars.secondary_stack_factor));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
