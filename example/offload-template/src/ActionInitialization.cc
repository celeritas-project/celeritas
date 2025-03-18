//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include <accel/TrackingManagerIntegration.hh>

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

namespace celeritas
{
namespace example
{
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
    // Set up Celeritas integration
    celeritas::TrackingManagerIntegration::Instance().BuildForMaster();

    // RunAction is responsible for initializing Celeritas
    this->SetUserAction(new RunAction());
}

//---------------------------------------------------------------------------//
/*!
 * Set up all worker thread user actions and Celeritas offload interface.
 */
void ActionInitialization::Build() const
{
    // Set up Celeritas integration
    celeritas::TrackingManagerIntegration::Instance().Build();

    // Initialize Geant4 user actions
    this->SetUserAction(new RunAction());
    this->SetUserAction(new PrimaryGeneratorAction());

    // Print diagnostics
    this->SetUserAction(new EventAction());
}

}  // namespace example
}  // namespace celeritas
