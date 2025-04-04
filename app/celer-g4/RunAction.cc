//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <G4RunManager.hh>
#include <G4StateManager.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/TimeOutput.hh"

#include "ExceptionHandler.hh"
#include "GeantDiagnostics.hh"
#include "GlobalSetup.hh"
#include "RootIO.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
RunAction::RunAction(SPConstOptions options,
                     SPParams params,
                     SPTransporter transport,
                     SPDiagnostics diagnostics,
                     bool init_shared)
    : options_{std::move(options)}
    , params_{std::move(params)}
    , transport_{std::move(transport)}
    , diagnostics_{std::move(diagnostics)}
    , init_shared_{init_shared}
{
    CELER_EXPECT(options_);
    CELER_EXPECT(params_);
    CELER_EXPECT(diagnostics_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(G4Run const* run)
{
    CELER_EXPECT(run);

    ExceptionConverter call_g4exception{"celer.init.global"};

    if (init_shared_)
    {
        // This worker (or master thread) is responsible for initializing
        // celeritas: initialize shared data and setup GPU on all threads
        CELER_TRY_HANDLE(params_->Initialize(*options_), call_g4exception);
        CELER_ASSERT(*params_);

        // Construct diagnostics
        CELER_TRY_HANDLE(diagnostics_->Initialize(*params_), call_g4exception);
        CELER_ASSERT(*diagnostics_);

        params_->timer()->RecordSetupTime(
            GlobalSetup::Instance()->GetSetupTime());
        get_transport_time_ = {};
    }
    else
    {
        CELER_TRY_HANDLE(params_->InitializeWorker(*options_),
                         call_g4exception);
    }

    if (transport_ && params_->mode() == SharedParams::Mode::enabled)
    {
        // Allocate data in shared thread-local transporter
        CELER_TRY_HANDLE(transport_->Initialize(*options_, *params_),
                         ExceptionConverter{"celer.init.local"});
        CELER_ASSERT(*transport_);
    }

    if (transport_)
    {
        // Set up local logger; "master" thread in MT already has
        // logging/exception set through celer-g4 main
        scoped_log_
            = std::make_unique<ScopedGeantLogger>(celeritas::self_logger());
        scoped_except_ = std::make_unique<ScopedGeantExceptionHandler>();
    }

    // Create a G4VExceptionHandler that dispatches to the shared
    // MultiException
    orig_eh_ = G4StateManager::GetStateManager()->GetExceptionHandler();
    static std::mutex exception_handle_mutex;
    exception_handler_ = std::make_shared<ExceptionHandler>(
        [meh = diagnostics_->multi_exception_handler()](std::exception_ptr ptr) {
            std::lock_guard scoped_lock{exception_handle_mutex};
            return (*meh)(ptr);
        },
        params_);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 *
 * \todo In the event of a failure during the stepping loop, it looks like the
 * exception manager might be unregistered by Geant4 at this point, leading to
 * a hard G4Exception crash.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    ExceptionConverter call_g4exception{"celer.finalize"};

    if (GlobalSetup::Instance()->root_sd_io())
    {
        // Close ROOT output of sensitive hits
        CELER_TRY_HANDLE(RootIO::Instance()->Close(), call_g4exception);
    }

    // Reset exception handler before finalizing diagnostics
    G4StateManager::GetStateManager()->SetExceptionHandler(orig_eh_);

    if (transport_ && *transport_)
    {
        params_->timer()->RecordActionTime(transport_->GetActionTime());
    }
    if (init_shared_)
    {
        params_->timer()->RecordTotalTime(get_transport_time_());
    }

    if (params_->mode() == SharedParams::Mode::enabled)
    {
        CELER_LOG(status) << "Finalizing Celeritas";

        if (transport_)
        {
            // Deallocate Celeritas state data (ensures that objects are
            // deleted on the thread in which they're created, necessary by
            // some geant4 thread-local allocators)
            CELER_TRY_HANDLE(transport_->Finalize(), call_g4exception);
        }
    }

    if (init_shared_)
    {
        // Finalize diagnostics (clearing exception handler) after most other
        // stuff can go wrong
        CELER_TRY_HANDLE(diagnostics_->Finalize(), call_g4exception);
        // Clear shared data (if any) and write output (if any)
        CELER_TRY_HANDLE(params_->Finalize(), call_g4exception);
    }

    scoped_log_.reset();
    scoped_except_.reset();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
