//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "accel/ExceptionConverter.hh"

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

    ExceptionConverter call_g4exception{"celer0001"};

    if (!SharedParams::CeleritasDisabled())
    {
        if (init_shared_)
        {
            // This worker (or master thread) is responsible for initializing
            // celeritas: initialize shared data and setup GPU on all threads
            // TODO: reusing the existing output registry is a hack needed to
            // preserve the GeantSimpleCalo output. This will be fixed in 0.5
            CELER_TRY_HANDLE(
                params_->Initialize(*options_, params_->output_reg()),
                call_g4exception);
            CELER_ASSERT(*params_);
        }
        else
        {
            CELER_TRY_HANDLE(SharedParams::InitializeWorker(*options_),
                             call_g4exception);
        }

        if (transport_)
        {
            // Allocate data in shared thread-local transporter
            CELER_TRY_HANDLE(transport_->Initialize(*options_, *params_),
                             call_g4exception);
            CELER_ASSERT(*transport_);
        }
    }

    if (init_shared_)
    {
        CELER_TRY_HANDLE(diagnostics_->Initialize(*params_), call_g4exception);
        CELER_ASSERT(*diagnostics_);

        diagnostics_->timer()->RecordSetupTime(
            GlobalSetup::Instance()->GetSetupTime());
        get_transport_time_ = {};
    }

    // Create a G4VExceptionHandler that dispatches to the shared
    // MultiException
    orig_eh_ = G4StateManager::GetStateManager()->GetExceptionHandler();
    static std::mutex exception_handle_mutex;
    exception_handler_ = std::make_shared<ExceptionHandler>(
        [meh = diagnostics_->multi_exception_handler()](std::exception_ptr ptr) {
            std::lock_guard scoped_lock{exception_handle_mutex};
            return (*meh)(ptr);
        });
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    ExceptionConverter call_g4exception{"celer0005"};

    if (GlobalSetup::Instance()->root_sd_io())
    {
        // Close ROOT output of sensitive hits
        CELER_TRY_HANDLE(RootIO::Instance()->Close(), call_g4exception);
    }

    // Reset exception handler before finalizing diagnostics
    G4StateManager::GetStateManager()->SetExceptionHandler(orig_eh_);

    if (transport_ && !SharedParams::CeleritasDisabled())
    {
        diagnostics_->timer()->RecordActionTime(transport_->GetActionTime());
    }
    if (init_shared_)
    {
        diagnostics_->timer()->RecordTotalTime(get_transport_time_());

        CELER_TRY_HANDLE(diagnostics_->Finalize(), call_g4exception);
    }

    if (!SharedParams::CeleritasDisabled())
    {
        CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

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
        // Clear shared data (if any) and write output (if any)
        CELER_TRY_HANDLE(params_->Finalize(), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
