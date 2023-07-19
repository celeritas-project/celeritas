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
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "accel/ExceptionConverter.hh"

#include "GlobalSetup.hh"
#include "HitRootIO.hh"

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
                     SPStepCounter step_counter,
                     bool init_celeritas,
                     bool init_diagnostics)
    : options_{std::move(options)}
    , params_{std::move(params)}
    , transport_{std::move(transport)}
    , step_counter_{std::move(step_counter)}
    , init_celeritas_{init_celeritas}
    , init_diagnostics_{init_diagnostics}
    , disable_offloading_(!celeritas::getenv("CELER_DISABLE").empty())
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

    ExceptionConverter call_g4exception{"celer0001"};

    if (!disable_offloading_)
    {
        if (init_celeritas_)
        {
            // This worker (or master thread) is responsible for initializing
            // celeritas
            if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
            {
                // To allow ORANGE to work for testing purposes, pass the GDML
                // input filename to Celeritas
                const_cast<SetupOptions&>(*options_).geometry_file
                    = GlobalSetup::Instance()->GetGeometryFile();
            }

            // Initialize shared data and setup GPU on all threads
            CELER_TRY_HANDLE(params_->Initialize(*options_), call_g4exception);
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
            CELER_ENSURE(*transport_);
        }
    }

    if (init_diagnostics_)
    {
        output_reg_ = *params_ ? params_->Params()->output_reg()
                               : std::make_shared<OutputRegistry>();

        // Create the track step diagnostic
        if (GlobalSetup::Instance()->CountTrackSteps())
        {
            size_type num_streams
                = *params_ ? params_->Params()->max_streams()
                           : get_num_threads(*G4RunManager::GetRunManager());

            CELER_TRY_HANDLE(
                step_counter_->Initialize(
                    GlobalSetup::Instance()->GetTrackStepBins(), num_streams),
                call_g4exception);
            CELER_ASSERT(*step_counter_);

            // Add to output interface
            CELER_TRY_HANDLE(output_reg_->insert(step_counter_),
                             call_g4exception);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    ExceptionConverter call_g4exception{"celer0005"};

    if (!disable_offloading_)
    {
        CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

        if (transport_)
        {
            // Deallocate Celeritas state data (ensures that objects are
            // deleted on the thread in which they're created, necessary by
            // some geant4 thread-local allocators)
            CELER_TRY_HANDLE(transport_->Finalize(), call_g4exception);
        }

        if (init_celeritas_)
        {
            // Clear shared data and write
            CELER_TRY_HANDLE(params_->Finalize(), call_g4exception);
        }
    }
    else if (init_diagnostics_)
    {
        // Write the diagnostic output if it wasn't done as part of the
        // Celeritas finalization
        // TODO: duplcated in SharedParams
        std::string output_file
            = GlobalSetup::Instance()->GetSetupOptions()->output_file;
        if (!output_file.empty())
        {
#if CELERITAS_USE_JSON
            CELER_LOG(info) << "Writing Geant4 diagnostic output to \""
                            << output_file << '"';

            std::ofstream outf(output_file);
            CELER_VALIDATE(outf,
                           << "failed to open output file at \"" << output_file
                           << '"');
            output_reg_->output(&outf);
#else
            CELER_LOG(warning)
                << "JSON support is not enabled, so no output will "
                   "be written to \""
                << output_file << '"';
#endif
        }
    }

    if (GlobalSetup::Instance()->GetWriteSDHits())
    {
        // Close ROOT output of sensitive hits
        CELER_TRY_HANDLE(HitRootIO::Instance()->Close(), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
