//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/io/Logger.hh"
#include "accel/LocalTransporter.hh"

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
ActionInitialization::ActionInitialization() : init_celeritas_{true}
{
    // Create params to be shared across worker threads
    params_ = std::make_shared<celeritas::SharedParams>();
    // Make global setup commands available to UI
    GlobalSetup::Instance();
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on the master thread.
 *
 * Since our \c RunAction::EndOfRunAction only calls \c SharedParams::Finalize
 * on the master thread, we need a special case for MT mode.
 */
void ActionInitialization::BuildForMaster() const
{
    CELER_LOG_LOCAL(status) << "Constructing user action on master thread";

    // Run action for 'master' has no track states and is responsible for
    // setting up celeritas
    this->SetUserAction(
        new RunAction{GlobalSetup::Instance()->GetSetupOptions(),
                      params_,
                      nullptr,
                      init_celeritas_});

    // Subsequent worker threads must not set up celeritas
    init_celeritas_ = false;
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

    // Run action sets up Celeritas (init_celeritas_ will be true iff
    // using a serial run manager)
    this->SetUserAction(
        new RunAction{GlobalSetup::Instance()->GetSetupOptions(),
                      params_,
                      transport,
                      init_celeritas_});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    this->SetUserAction(new EventAction{params_, transport});
    // Tracking action offloads tracks to device and kills them
    this->SetUserAction(new TrackingAction{params_, transport});
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
