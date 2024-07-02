//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/optical/Types.hh"

#include "OpticalPrimary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
using OpticalValueGrid = GenericGridData;
using OpticalValueGridId = OpaqueId<OpticalValueGrid>;

//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical physics.
 */
struct OpticalPhysicsParamsScalars
{
    //! Offset to create an ActionId from a ModelId
    ActionId::size_type model_to_action{};
    //! Number of optical physics models
    OpticalModelId::size_type num_models{};

    //! Secondary storage per state size
    real_type secondary_stack_factor = 2;

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model_to_action >= 1
               && num_models > 0
               && secondary_stack_factor >= 0;
    }

    //! Indicate a discrete interaction was rejected by the integral method
    CELER_FORCEINLINE_FUNCTION ActionId discrete_action() const
    {
        return ActionId{model_to_action - 1};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared optical physics data
 */
template<Ownership W, MemSpace M>
struct OpticalPhysicsParamsData
{
    //!@{
    //! \name Type aliases
    template <class T>
    using Items = Collection<T, W, M>;
    template <class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;
    using OpticalModelMfp = ItemMap<OpticalModelId, OpticalValueGridId>;
    //!@}



    //// DATA ////
    Items<OpticalValueGrid> grids;
    OpticalMaterialItems<OpticalModelMfp> mat_model_mfp;

    // Backend storage
    Items<real_type> reals;
    

    // Non-templated data
    OpticalPhysicsParamsScalars scalars;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !mat_model_mfp.empty() && scalars;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalPhysicsParamsData& operator=(OpticalPhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        reals = other.reals;
        grids = other.grids;
        mat_model_mfp = other.mat_model_mfp;

        return *this;
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Optical physics state data for a single track.
 *
 * State that's persistent across steps:
 * - Remaining number of mean free paths to the next discrete interaction
 *
 * State that is reset at every step:
 * - Current macroscopic cross section
 * - Secondaries emitted from an interaction
 */
struct OpticalPhysicsTrackState
{
    real_type interaction_mfp;  //!< Remaining MFP to interaction

    // TEMPORARY STATE
    real_type macro_xs;  //!< Total cross section for discrete interactions
    Span<OpticalPrimary> secondaries;  //!< Emitted secondaries
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an optical physics track state.
 */
struct OpticalPhysicsTrackInitializer
{
};

//---------------------------------------------------------------------------//
/*!
 * Dynamic optical physics state data.
 */
template<Ownership W, MemSpace M>
struct OpticalPhysicsData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    
    StateItems<OpticalPhysicsTrackState> state;  //!< Track state [track]
    StackAllocatorData<OpticalPrimary, W, M> secondaries;  //!< Secondary stack
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
