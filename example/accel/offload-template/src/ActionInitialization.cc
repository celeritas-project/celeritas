//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "Celeritas.hh"
#include "EventAction.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitialization::ActionInitialization() : G4VUserActionInitialization() {}

//---------------------------------------------------------------------------//
/*!
 * Set up all user actions and Celeritas' offloading interface.
 */
void ActionInitialization::Build() const
{
    // Construct Celeritas offloading interface
    CelerSimpleOffload().Build(
        &CelerSetupOptions(), &CelerSharedParams(), &CelerLocalTransporter());

    // Add Celeritas tracking manager to electrons, positrons, and gammas
    auto* celer_tracking = new celeritas::TrackingManagerOffload(
        &CelerSharedParams(), &CelerLocalTransporter());
    G4Electron::Definition()->SetTrackingManager(celer_tracking);
    G4Positron::Definition()->SetTrackingManager(celer_tracking);
    G4Gamma::Definition()->SetTrackingManager(celer_tracking);

    // Initialize Geant4 user actions
    this->SetUserAction(new RunAction());
    this->SetUserAction(new EventAction());
    this->SetUserAction(new PrimaryGeneratorAction());
}
