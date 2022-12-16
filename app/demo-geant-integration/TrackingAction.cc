//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <G4ParticleDefinition.hh>
#include <G4Track.hh>
#include <G4TrackStatus.hh>

#include "corecel/Assert.hh"

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
    CELER_EXPECT(*params_);
    CELER_EXPECT(*transport_);

    if (transport_->TryOffload(*track))
    {
        // Celeritas is transporting this track
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
