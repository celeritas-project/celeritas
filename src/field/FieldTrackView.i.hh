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
// Construct from GeoTrackView and ParticleTrackView
CELER_FUNCTION
FieldTrackView::FieldTrackView(const GeoTrackView&      geo_view,
                               const ParticleTrackView& particle_view)
    : on_boundary_(false)
    , charge_(particle_view.charge())
    , step_(geo_view.next_step())
    , safety_(0)
    , origin_{0, 0, 0}
    , vgstate_(geo_view.vgstate())
    , vgnext_(geo_view.vgnext())
{
    state_.pos = geo_view.pos();
    axpy(particle_view.momentum().value(), geo_view.dir(), &state_.mom);

    navigator_ = vgstate_.Top()->GetLogicalVolume()->GetNavigator();
    CELER_ASSERT(navigator_);
}

//---------------------------------------------------------------------------//
/*
 * vecgeom VNavigator associated with the current navigation state
 */
CELER_FUNCTION real_type FieldTrackView::linear_propagator(Real3     pos,
                                                           Real3     dir,
                                                           real_type step)
{
    CELER_ASSERT(navigator_);
    return navigator_->ComputeStepAndSafetyAndPropagatedState(
        detail::to_vector(pos),
        detail::to_vector(dir),
        step,
        vgstate_,
        vgnext_,
        true,
        safety_);
}

//---------------------------------------------------------------------------//
/*!
 * Update vecgeom states for a geometry limited step
 */
CELER_FUNCTION void FieldTrackView::update_vgstates()
{
    vgstate_ = vgnext_;
    vgstate_.SetBoundaryState(true);
    vgnext_.Clear();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
