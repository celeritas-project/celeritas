//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/io/Logger.hh"
#include "accel/LocalTransporter.hh"

#include "EventAction.hh"
#include "GlobalSetup.hh"
#include "HepMC3PrimaryGeneratorAction.hh"
#include "PGPrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct global data to be shared across Celeritas workers.
 */
ActionInitialization::ActionInitialization()
    : init_celeritas_{true}, init_diagnostics_{true}
{
    // Create params to be shared across worker threads
    params_ = std::make_shared<SharedParams>();
    // Create Geant4 diagnostics to be shared across worker threads
    diagnostics_ = std::make_shared<GeantDiagnostics>();
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
                      diagnostics_,
                      init_celeritas_,
                      init_diagnostics_});

    // Subsequent worker threads must not set up celeritas or diagnostics
    init_celeritas_ = false;
    init_diagnostics_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(status) << "Constructing user actions on worker threads";

    // Primary generator emits source particles
    if (!GlobalSetup::Instance()->GetEventFile().empty())
    {
        this->SetUserAction(new HepMC3PrimaryGeneratorAction());
    }
    else
    {
        this->SetUserAction(new PGPrimaryGeneratorAction());
    }

    // Create thread-local transporter to share between actions
    auto transport = std::make_shared<LocalTransporter>();

    // Run action sets up Celeritas (init_celeritas_ will be true iff
    // using a serial run manager)
    this->SetUserAction(
        new RunAction{GlobalSetup::Instance()->GetSetupOptions(),
                      params_,
                      transport,
                      diagnostics_,
                      init_celeritas_,
                      init_diagnostics_});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    this->SetUserAction(new EventAction{params_, transport});
    // Tracking action offloads tracks to device and kills them
    this->SetUserAction(new TrackingAction{params_, transport, diagnostics_});
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
