//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <G4RunManager.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/GeantSetup.hh"

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

    if (!SharedParams::CeleritasDisabled())
    {
        if (init_shared_)
        {
            // This worker (or master thread) is responsible for initializing
            // celeritas: initialize shared data and setup GPU on all threads
            params_->Initialize(*options_);
            CELER_ASSERT(*params_);
        }
        else
        {
            SharedParams::InitializeWorker(*options_);
        }

        if (transport_)
        {
            // Allocate data in shared thread-local transporter
            transport_->Initialize(*options_, *params_);
            CELER_ASSERT(*transport_);
        }
    }

    if (init_shared_)
    {
        diagnostics_->Initialize(*params_);
        CELER_ASSERT(*diagnostics_);

        diagnostics_->Timer()->RecordSetupTime(
            GlobalSetup::Instance()->GetSetupTime());
        get_transport_time_ = {};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    if (RootIO::use_root())
    {
        // Close ROOT output of sensitive hits
        RootIO::Instance()->Close();
    }

    if (transport_ && !SharedParams::CeleritasDisabled())
    {
        diagnostics_->Timer()->RecordActionTime(transport_->GetActionTime());
    }
    if (init_shared_)
    {
        diagnostics_->Timer()->RecordTotalTime(get_transport_time_());
        diagnostics_->Finalize();
    }

    if (!SharedParams::CeleritasDisabled())
    {
        CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

        if (transport_)
        {
            // Deallocate Celeritas state data (ensures that objects are
            // deleted on the thread in which they're created, necessary by
            // some geant4 thread-local allocators)
            transport_->Finalize();
        }
    }

    if (init_shared_)
    {
        // Clear shared data (if any) and write output (if any)
        params_->Finalize();
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
