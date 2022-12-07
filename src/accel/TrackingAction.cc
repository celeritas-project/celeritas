//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <G4Track.hh>

#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    CELER_LOG_LOCAL(debug) << "TrackingAction::PreUserTrackingAction";

    // Check against list of supported celeritas particles
    if (false)
    {
        // TODO: send to transporter
        const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
