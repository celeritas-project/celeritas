//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"

#include "GlobalSetup.hh"
#include "HitRootIO.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
RunAction::RunAction(SPConstOptions options,
                     SPParams params,
                     SPTransporter transport,
                     bool init_celeritas)
    : options_{std::move(options)}
    , params_{std::move(params)}
    , transport_{std::move(transport)}
    , init_celeritas_{init_celeritas}
{
    CELER_EXPECT(options_);
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(G4Run const* run)
{
    CELER_EXPECT(run);

    celeritas::ExceptionConverter call_g4exception{"celer0001"};

    if (init_celeritas_)
    {
        // This worker (or master thread) is responsible for initializing
        // celeritas
        if (CELERITAS_GEO == CELERITAS_GEO_ORANGE)
        {
            // To allow ORANGE to work for testing purposes, pass the GDML
            // input filename to Celeritas
            const_cast<celeritas::SetupOptions&>(*options_).geometry_file
                = GlobalSetup::Instance()->GetGeometryFile();
        }

        // Initialize shared data and setup GPU on all threads
        CELER_TRY_HANDLE(params_->Initialize(*options_), call_g4exception);
        CELER_ASSERT(*params_);
    }
    else
    {
        CELER_TRY_HANDLE(celeritas::SharedParams::InitializeWorker(*options_),
                         call_g4exception);
    }

    if (transport_)
    {
        // Allocate data in shared thread-local transporter
        CELER_TRY_HANDLE(transport_->Initialize(*options_, *params_),
                         call_g4exception);
        CELER_ENSURE(*transport_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";
    celeritas::ExceptionConverter call_g4exception{"celer0005"};

    if (transport_)
    {
        // Deallocate Celeritas state data (ensures that objects are deleted on
        // the thread in which they're created, necessary by some geant4
        // thread-local allocators)
        CELER_TRY_HANDLE(transport_->Finalize(), call_g4exception);
    }

    if (init_celeritas_)
    {
        // Clear shared data and write
        CELER_TRY_HANDLE(params_->Finalize(), call_g4exception);
    }

    if (GlobalSetup::Instance()->GetWriteSDHits())
    {
        // Close ROOT output of sensitive hits
        CELER_TRY_HANDLE(HitRootIO::Instance()->Close(), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace demo_geant
