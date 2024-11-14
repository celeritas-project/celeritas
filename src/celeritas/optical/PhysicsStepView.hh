//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsStepView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "PhysicsData.hh"
#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access step-local (non-persistent) optical physics track data.
 */
class PhysicsStepView
{
  public:
    //!@{
    //! \name Type aliases
    using PhysicsParamsRef = NativeCRef<PhysicsParamsData>;
    using PhysicsStateRef = NativeRef<PhysicsStateData>;
    //!@}

  public:
    // Construct from state data for a given track
    inline CELER_FUNCTION PhysicsStepView(PhysicsParamsRef const&,
                                          PhysicsStateRef const&,
                                          TrackSlotId);

    //// Cross section scrach space ////

    // Set cross section for a given model
    inline CELER_FUNCTION real_type& per_model_xs(ModelId mid);

    // Set total cross section
    inline CELER_FUNCTION void macro_xs(real_type xs);

    // Retrieve cross section for a given model
    inline CELER_FUNCTION real_type per_model_xs(ModelId mid) const;

    // Retrieve total cross section
    inline CELER_FUNCTION real_type macro_xs() const;

  private:
    PhysicsParamsRef const& params_;
    PhysicsStateRef const& state_;
    TrackSlotId track_id_;

    ItemId<real_type> per_model_xs_id(ModelId) const;

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION PhysicsTrackState const& state() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state data for a given track.
 */
CELER_FUNCTION
PhysicsStepView::PhysicsStepView(PhysicsParamsRef const& params,
                                 PhysicsStateRef const& state,
                                 TrackSlotId track)
    : params_(params), state_(state), track_id_(track)
{
    CELER_EXPECT(track_id_ < state_.states.size());
}

//---------------------------------------------------------------------------//
/*!
 * Set cross section for a given model.
 */
CELER_FUNCTION real_type& PhysicsStepView::per_model_xs(ModelId model)
{
    return state_.per_model_xs[this->per_model_xs_id(model)];
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve cross section for a given model.
 */
CELER_FUNCTION real_type PhysicsStepView::per_model_xs(ModelId model) const
{
    return state_.per_model_xs[this->per_model_xs_id(model)];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the index for per model cross section scratch space.
 */
CELER_FUNCTION ItemId<real_type>
PhysicsStepView::per_model_xs_id(ModelId model) const
{
    CELER_EXPECT(model < params_.scalars.num_models);
    size_type idx = track_id_.get() * params_.scalars.num_models + model.get();
    CELER_EXPECT(idx < state_.per_model_xs.size());
    return ItemId<real_type>{idx};
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve total cross section.
 */
CELER_FUNCTION real_type PhysicsStepView::macro_xs() const
{
    return this->state().macro_xs;
}

//---------------------------------------------------------------------------//
/*!
 * Set total cross section.
 */
CELER_FUNCTION void PhysicsStepView::macro_xs(real_type xs)
{
    this->state().macro_xs = xs;
}

//---------------------------------------------------------------------------//
//! Access the state associated with the track
CELER_FUNCTION PhysicsTrackState& PhysicsStepView::state()
{
    return state_.states[track_id_];
}

//! Access the state associated with the track
CELER_FUNCTION PhysicsTrackState const& PhysicsStepView::state() const
{
    return state_.states[track_id_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
