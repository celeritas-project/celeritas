//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
using ValueGrid = GenericGridRecord;
using ValueGridId = OpaqueId<ValueGrid>;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Set of value grids for all optical materials.
 */
struct ValueTable
{
    ItemRange<ValueGridId> grids;  //!< Value grid by optical material index

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const { return !grids.empty(); }
};

using ValueTableId = OpaqueId<ValueTable>;

struct ModelTables
{
    ValueTableId mfp_table;

    //! Whether data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(mfp_table);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical physics.
 */
struct PhysicsParamsScalars
{
    //! Offset to create an ActionId from a ModelId
    ActionId::size_type model_to_action{};

    //! Number of optical physics models
    ModelId::size_type num_models{};

    //! Secondary storage per state size
    real_type secondary_stack_factor = 2;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return model_to_action >= 1 && num_models > 0
               && secondary_stack_factor >= 0;
    }

    //! Indicate a disrete interaction was rejected by the integral method
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
    {
        return ActionId{model_to_action - 1};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared optical physics data.
 */
template<Ownership W, MemSpace M>
struct PhysicsParamsData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ModelItems = Collection<T, W, M, ModelId>;
    //!@}

    //// MEMBER DATA ////

    Items<ValueGrid> grids;
    Items<ValueGridId> grid_ids;
    Items<ValueTable> tables;
    ModelItems<ModelTables> model_tables;

    // Backend data
    Items<real_type> reals;

    // Non-templated data
    PhysicsParamsScalars scalars;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !grids.empty() && !grid_ids.empty() && !tables.empty()
               && !model_tables.empty() && !reals.empty() && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsParamsData<W, M>& operator=(PhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        grids = other.grids;
        grid_ids = other.grid_ids;
        tables = other.tables;
        reals = other.reals;
        scalars = other.scalars;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Physics state data for a single track.
 */
struct PhysicsTrackState
{
    // PERSISTENT STATE

    real_type interaction_mfp;

    // TEMPORARY STATE
    real_type macro_xs;  //!< Total macroscopic cross section [len^-1]
    real_type energy_deposition;  //!< local energy deposition in a step
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an optical physics track state.
 */
struct PhysicsTrackInitializer
{
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic optical physics state data.
 */
template<Ownership W, MemSpace M>
struct PhysicsStateData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    //!@}

    //// DATA ////

    StateItems<PhysicsTrackState> states;

    //// METHODS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !states.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return states.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    PhysicsStateData<W, M>& operator=(PhysicsStateData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        states = other.states;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>* state, size_type size)
{
    CELER_ASSERT(state);
    CELER_EXPECT(size > 0);

    resize(&state->states, size);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
