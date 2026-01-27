//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepView.cc
//---------------------------------------------------------------------------//
#include "GeantStepView.hh"

#include "GeantTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Update track from step data.
 *
 * Copies step length and step point data to the track. This is similar to
 * \c G4Step::UpdateTrack but applies only to attributes we know about and
 * safely handles null pointers.
 */
void GeantStepView::update_track()
{
    CELER_EXPECT(step_.GetTrack());

    GeantTrackView track{*step_.GetTrack()};
    GeantParticleView particle_view = track.particle();

    // Update pre-step point if present
    if (G4StepPoint* pre_step = step_.GetPreStepPoint())
    {
        GeantStepPointView{*pre_step}.update_from_particle(particle_view);
        track.track().SetTouchableHandle(pre_step->GetTouchableHandle());
    }

    // Update post-step point and track from post-step if present
    if (G4StepPoint* post_step = step_.GetPostStepPoint())
    {
        GeantStepPointView post_view{*post_step};
        post_view.update_from_particle(particle_view);

        // Copy post-step state to track
        track.time(post_view.time());
        track.pos(post_view.pos());
        track.energy(post_view.energy());
        track.dir(post_view.dir());
        track.weight(post_view.weight());

        track.track().SetNextTouchableHandle(post_step->GetTouchableHandle());
        track.track().SetVelocity(post_step->GetVelocity());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
