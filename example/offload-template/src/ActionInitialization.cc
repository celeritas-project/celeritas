//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "Celeritas.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitialization::ActionInitialization() : G4VUserActionInitialization() {}

//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas offload on master thread and initialize it via the
 * \c G4UserRunAction .
 */
void ActionInitialization::BuildForMaster() const
{
    // Construct Celeritas offloading interface on master thread
    CelerSimpleOffload().BuildForMaster(&CelerSetupOptions(),
                                        &CelerSharedParams());

    // RunAction is responsible for initializing Celeritas
    this->SetUserAction(new RunAction());
}

//---------------------------------------------------------------------------//
/*!
 * Set up all worker thread user actions and Celeritas offload interface.
 */
void ActionInitialization::Build() const
{
    // Construct Celeritas offloading interface on worker thread
    CelerSimpleOffload().Build(
        &CelerSetupOptions(), &CelerSharedParams(), &CelerLocalTransporter());

    // Initialize Geant4 user actions
    this->SetUserAction(new RunAction());
    this->SetUserAction(new EventAction());
    this->SetUserAction(new PrimaryGeneratorAction());
}
