//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <G4ParticleDefinition.hh>
#include <G4Track.hh>
#include <G4TrackStatus.hh>

#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas shared data.
 */
TrackingAction::TrackingAction(SPParams params, SPTransporter transport)
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
    // Check against list of supported celeritas particles
    PDGNumber pdg{track->GetDefinition()->GetPDGEncoding()};
    if (params_->params->particle()->find(pdg))
    {
        // Send track to transporter and kill
        transport_->add(*track);
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
