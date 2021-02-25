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
    : vgstate_(geo_view.vgstate()), vgnext_(geo_view.vgnext())
{
    h_ = geo_view.next_step();
    m_ = particle_view.mass();
    q_ = particle_view.charge();

    for (size_type i = 0; i < 3; ++i)
    {
        y_[i]     = geo_view.pos()[i];
        y_[i + 3] = (particle_view.momentum().value()) * geo_view.dir()[i];
    }

    safety_ = 0;
    origin_ = {0, 0, 0};

    navigator_ = vgstate_.Top()->GetLogicalVolume()->GetNavigator();
    CELER_ASSERT(navigator_);
}

//---------------------------------------------------------------------------//
/*!
 * momentum squre
 */
CELER_FUNCTION real_type FieldTrackView::momentum_squre() const
{
    real_type value = y_[3] * y_[3] + y_[4] * y_[4] + y_[5] * y_[5];
    CELER_ASSERT(value > 0);
    return value;
}
//---------------------------------------------------------------------------//
/*
 * vecgeom VNavigator associated with the current navigation state
 */
CELER_FUNCTION real_type FieldTrackView::linear_propagator(vec3_type pos,
                                                           vec3_type dir,
                                                           real_type step)
{
    CELER_ASSERT(navigator_);
    return navigator_->ComputeStepAndSafetyAndPropagatedState(
        pos, dir, step, vgstate_, vgnext_, true, safety_);
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

} // namespace celeritas
