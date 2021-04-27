//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTrackView.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from GeoTrackView and ParticleTrackView
 */
CELER_FUNCTION
FieldTrackView::FieldTrackView(const GeoTrackView&      geo_view,
                               const ParticleTrackView& particle_view)
    : on_boundary_(false)
    , charge_(particle_view.charge())
    , step_(0)
    , safety_(0)
    , vgstate_(geo_view.vgstate())
    , vgnext_(geo_view.vgnext())
{
    state_.pos = geo_view.pos();
    axpy(particle_view.momentum().value(), geo_view.dir(), &state_.mom);

    navigator_ = vgstate_.Top()->GetLogicalVolume()->GetNavigator();
    CELER_ASSERT(navigator_);

    // Initialize safety at the current position
    this->update_safety(state_.pos);
}

//---------------------------------------------------------------------------//
/*!
 * Update the estimated safety at a given position
 */
CELER_FUNCTION void FieldTrackView::update_safety(Real3 pos)
{
    safety_ = navigator_->GetSafetyEstimator()->ComputeSafety(
        detail::to_vector(pos), vgstate_);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the linear step length to the next boundary from a given position
 * and a direction, and update the safety without updating vegeom states
 */
CELER_FUNCTION real_type FieldTrackView::compute_step(Real3 pos, Real3 dir)
{
    CELER_ASSERT(navigator_);
    return navigator_->ComputeStepAndSafety(detail::to_vector(pos),
                                            detail::to_vector(dir),
                                            vecgeom::kInfLength,
                                            vgstate_,
                                            true,
                                            safety_,
                                            false);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate states to the next boundary from a given position and a direction
 * and update satety for a geometry limited step
 */
CELER_FUNCTION void FieldTrackView::linear_propagator(Real3 pos, Real3 dir)
{
    navigator_->ComputeStepAndPropagatedState(detail::to_vector(pos),
                                              detail::to_vector(dir),
                                              vecgeom::kInfLength,
                                              vgstate_,
                                              vgnext_);

    safety_  = 0;
    vgstate_ = vgnext_;
    vgstate_.SetBoundaryState(true);
    vgnext_.Clear();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
