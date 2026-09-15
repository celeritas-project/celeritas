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
    CELER_EXPECT(s_.GetTrack());

    GeantTrackView track{*s_.GetTrack()};
    GeantParticleView particle_view = track.particle();

    // Update pre-step point if present
    if (G4StepPoint* pre_step = s_.GetPreStepPoint())
    {
        GeantStepPointView{*pre_step}.update_from_particle(particle_view);
        track.track().SetTouchableHandle(pre_step->GetTouchableHandle());
    }

    // Update post-step point and track from post-step if present
    if (G4StepPoint* post_step = s_.GetPostStepPoint())
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
/*!
 * Delete a step point.
 *
 * This sets the step point to null if supported by the Geant4 version,
 * otherwise does nothing (no reset available before v11.0.3).
 */
void GeantStepView::delete_step_point(StepPoint sp)
{
    CELER_EXPECT(sp != StepPoint::size_);

#if G4VERSION_NUMBER >= 1103
    if (sp == StepPoint::pre)
    {
        s_.ResetPreStepPoint(nullptr);
    }
    else
    {
        s_.ResetPostStepPoint(nullptr);
    }
#else
    // No "reset" available before v11.0.3
    CELER_DISCARD(sp);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
