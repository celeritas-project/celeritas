//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include <G4RunManager.hh>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"

#include "EventAction.hh"
#include "GlobalSetup.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct global data to be shared across Celeritas workers.
 */
ActionInitialization::ActionInitialization()
{
    // Create params to be shared across worker threads
    params_ = celeritas::SharedParams::MakeShared();
    // Make global setup commands available to UI
    GlobalSetup::Instance();
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(status) << "Constructing user actions on worker threads";

    // Primary generator emits source particles
    this->SetUserAction(new PrimaryGeneratorAction());

    // Create thread-local transporter to share between actions
    auto transport = std::make_shared<celeritas::LocalTransporter>();

    // Run action sets up Celeritas
    this->SetUserAction(new RunAction{
        GlobalSetup::Instance()->GetSetupOptions(), params_, transport});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    this->SetUserAction(new EventAction{transport});
    // Tracking action offloads tracks to device and kills them
    this->SetUserAction(new TrackingAction{params_, transport});
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
