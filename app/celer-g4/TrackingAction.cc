//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "accel/ExceptionConverter.hh"

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
    CELER_EXPECT((params_->mode() == SharedParams::Mode::enabled)
                 == static_cast<bool>(*transport_));

    auto const mode = params_->mode();
    if (mode == SharedParams::Mode::disabled)
        return;

    auto const& particles = params_->OffloadParticles();
    if (std::find(particles.begin(), particles.end(), track->GetDefinition())
        != particles.end())
    {
        if (mode == SharedParams::Mode::enabled)
        {
            // Celeritas is transporting this track
            ExceptionConverter call_g4exception{"celer.track.push",
                                                params_.get()};
            CELER_TRY_HANDLE(transport_->Push(const_cast<G4Track&>(*track)),
                             call_g4exception);
        }
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Collect diagnostic data at the end of a track.
 */
void TrackingAction::PostUserTrackingAction(G4Track const* track)
{
    if (diagnostics_->step_diagnostic())
    {
        diagnostics_->step_diagnostic()->Update(track);
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
