//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <G4Track.hh>
#include <G4TrackStatus.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas shared and thread-local data.
 */
TrackingAction::TrackingAction(SPConstParams params,
                               SPTransporter transport,
                               SPDiagnostics diagnostics)
    : params_(params), transport_(transport), diagnostics_(diagnostics)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
    CELER_EXPECT(diagnostics_);
}

//---------------------------------------------------------------------------//
/*!
 * At the start of a track, determine whether to use Celeritas to transport it.
 *
 * If the track is one of a few predetermined EM particles, we pass it to
 * Celeritas (which queues the track on its buffer and potentially flushes it)
 * and kill the Geant4 track.
 */
void TrackingAction::PreUserTrackingAction(G4Track const* track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(static_cast<bool>(*params_)
                 == !SharedParams::CeleritasDisabled());
    CELER_EXPECT(static_cast<bool>(*params_) == static_cast<bool>(*transport_));

    if (SharedParams::CeleritasDisabled())
        return;

    auto const& allowed_particles = params_->OffloadParticles();
    if (std::find(std::begin(allowed_particles),
                  std::end(allowed_particles),
                  track->GetDefinition())
        != std::end(allowed_particles))
    {
        // Celeritas is transporting this track
        transport_->Push(*track);
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Collect diagnostic data at the end of a track.
 */
void TrackingAction::PostUserTrackingAction(G4Track const* track)
{
    if (diagnostics_->StepDiagnostic())
    {
        diagnostics_->StepDiagnostic()->Update(track);
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
