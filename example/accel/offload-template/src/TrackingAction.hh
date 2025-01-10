//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/TrackingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4UserTrackingAction.hh>

//---------------------------------------------------------------------------//
/*!
 * Generate primaries.
 */
class TrackingAction : public G4UserTrackingAction
{
  public:
    // Construct empty
    TrackingAction();

    // Stop and kill particles in Geant4; offload state to Celeritas
    void PreUserTrackingAction(G4Track const* track) final;
};
