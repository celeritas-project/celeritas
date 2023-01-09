//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4ParticleDefinition.hh>
#include <G4Positron.hh>
#include <G4Track.hh>
#include <G4TrackStatus.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "accel/ExceptionConverter.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas shared data.
 */
TrackingAction::TrackingAction(SPConstParams params, SPTransporter transport)
    : params_(params), transport_(transport)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
}

//---------------------------------------------------------------------------//
/*!
 * At the start of a track, determine whether to use Celeritas to transport it.
 */
void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    CELER_EXPECT(track);
    CELER_EXPECT(*params_);
    CELER_EXPECT(*transport_);

    static G4ParticleDefinition const* const allowed_particles[] = {
        G4Gamma::Gamma(),
        G4Electron::Electron(),
        G4Positron::Positron(),
    };

    if (std::find(std::begin(allowed_particles),
                  std::end(allowed_particles),
                  track->GetDefinition())
        != std::end(allowed_particles))
    {
        // Celeritas is transporting this track
        celeritas::ExceptionConverter call_g4exception{"celer0003"};
        CELER_TRY_HANDLE(transport_->Push(*track), call_g4exception);
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
