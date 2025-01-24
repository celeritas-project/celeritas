//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/HepMC3PrimaryGenerator.hh"
#include "accel/LocalTransporter.hh"

#include "EventAction.hh"
#include "GeantDiagnostics.hh"
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
 *
 * The parameters will be distributed to worker threads and all the actions.
 */
ActionInitialization::ActionInitialization(SPParams params)
    : params_{std::move(params)}
{
    CELER_EXPECT(params_);

    // Create Geant4 diagnostics to be shared across worker threads
    diagnostics_ = std::make_shared<GeantDiagnostics>();

    auto const& input = GlobalSetup::Instance()->input();
    if (!input.event_file.empty())
    {
        ExceptionConverter call_g4exception{"celer0007"};
        CELER_TRY_HANDLE(hepmc_gen_ = std::make_shared<HepMC3PrimaryGenerator>(
                             input.event_file),
                         call_g4exception);
        num_events_ = hepmc_gen_->NumEvents();
    }
    else
    {
        num_events_ = input.primary_options.num_events;
    }

    CELER_ENSURE(num_events_ > 0);
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
                      init_shared_});

    // Subsequent worker threads must not set up celeritas or diagnostics
    init_shared_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Construct actions on each worker thread.
 */
void ActionInitialization::Build() const
{
    CELER_LOG_LOCAL(status) << "Constructing user action";

    // Primary generator emits source particles
    std::unique_ptr<G4VUserPrimaryGeneratorAction> generator_action;
    if (hepmc_gen_)
    {
        ExceptionConverter call_g4exception{"celer0007"};
        CELER_TRY_HANDLE(
            generator_action
            = std::make_unique<HepMC3PrimaryGeneratorAction>(hepmc_gen_),
            call_g4exception);
    }
    else
    {
        ExceptionConverter call_g4exception{"celer0006"};
        CELER_TRY_HANDLE(
            generator_action = std::make_unique<PGPrimaryGeneratorAction>(
                GlobalSetup::Instance()->input().primary_options),
            call_g4exception);
    }
    this->SetUserAction(generator_action.release());

    // Create thread-local transporter to share between actions
    auto transport = std::make_shared<LocalTransporter>();

    // Run action sets up Celeritas (init_shared_ will be true if and only if
    // using a serial run manager)
    this->SetUserAction(
        new RunAction{GlobalSetup::Instance()->GetSetupOptions(),
                      params_,
                      transport,
                      diagnostics_,
                      init_shared_});
    // Event action saves event ID for offloading and runs queued particles at
    // end of event
    this->SetUserAction(new EventAction{params_, transport, diagnostics_});
    // Tracking action offloads tracks to device and kills them
    this->SetUserAction(new TrackingAction{params_, transport, diagnostics_});
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
