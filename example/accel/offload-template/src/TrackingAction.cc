//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/TrackingAction.cc
//---------------------------------------------------------------------------//
#include "TrackingAction.hh"

#include <accel/ExceptionConverter.hh>

#include "Celeritas.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "G4Track.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
TrackingAction::TrackingAction() : G4UserTrackingAction() {}

//---------------------------------------------------------------------------//
/*!
 * Offload available particles to be tracked in Celeritas.
 *
 * See \c src/accel/SharedParams::OffloadParticles() for list of particles.
 */
void TrackingAction::PreUserTrackingAction(G4Track const* track)
{
    CelerSimpleOffload().PreUserTrackingAction(const_cast<G4Track*>(track));
}
