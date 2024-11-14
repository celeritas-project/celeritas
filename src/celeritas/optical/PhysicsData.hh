//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ValueGrid = GenericGridRecord;
using ValueGridId = OpaqueId<ValueGrid>;
using ValueTable = ItemRange<ValueGrid>;
using ValueTableId = OpaqueId<ValueTable>;

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical physics.
 */
struct PhysicsParamsScalars
{
    //! Number of optical models
    ModelId::size_type num_models{};

    //! Offset to create an ActionId from a ModelId
    ActionId::size_type model_to_action{};

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return num_models > 0 && model_to_action >= 1;
    }

    //! Indicate a discrete interaction was rejected by integral method
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

    //! Non-templated data
    PhysicsParamsScalars scalars;

    //! Optical model data
    Items<ValueGrid> grids;
    ModelItems<ValueTable> mfp_tables;

    //! Backend storage
    Items<real_type> reals;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars) && !grids.empty()
               && !mfp_tables.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsParamsData<W, M>& operator=(PhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        this->scalars = other.scalars;
        this->grids = other.grids;
        this->mfp_tables = other.mfp_tables;
        this->reals = other.reals;
        return *this;
    }
};

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

    //// Data ////

    StateItems<PhysicsTrackState> states;
    Items<real_type> per_model_xs;  //!< XS [track][model]

    //// Methods ////

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !states.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return states.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsStateData<W, M>& operator=(PhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        states = other.states;
        per_model_xs = other.per_model_xs;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>* state,
                   HostCRef<PhysicsParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->states, size);
    resize(&state->per_model_xs, params.scalars.num_models * size);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
