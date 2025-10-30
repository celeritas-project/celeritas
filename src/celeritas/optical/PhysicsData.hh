//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ValueGrid = NonuniformGridRecord;
using ValueGridId = OpaqueId<ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical physics.
 */
struct PhysicsParamsScalars
{
    //! Number of optical models
    ModelId::size_type num_models{};

    //! Number of optical materials
    OptMatId::size_type num_materials{};

    //! Offset to create an ActionId from a ModelId
    ActionId first_model_action{};

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return num_models > 0 && num_materials > 0
               && first_model_action >= ActionId{1};
    }

    //! Undergo a discrete interaction
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
    {
        return first_model_action - 1;
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

    //! Backend storage
    Items<real_type> reals;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars) && !grids.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsParamsData<W, M>& operator=(PhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        this->scalars = other.scalars;
        this->grids = other.grids;
        this->reals = other.reals;
        return *this;
    }
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

    //// Persistent State Data ////

    StateItems<real_type> interaction_mfp;

    //// Temporary State Data ////

    StateItems<real_type> macro_xs;  //! < Total macroscopic cross section

    //// Methods ////

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !interaction_mfp.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return interaction_mfp.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    PhysicsStateData<W, M>& operator=(PhysicsStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        interaction_mfp = other.interaction_mfp;
        macro_xs = other.macro_xs;
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
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);

    resize(&state->interaction_mfp, size);
    resize(&state->macro_xs, size);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
