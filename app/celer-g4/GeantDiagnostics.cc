//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantDiagnostics.cc
//---------------------------------------------------------------------------//
#include "GeantDiagnostics.hh"

#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/user/StepDiagnostic.hh"

#include "GlobalSetup.hh"

#if CELERITAS_USE_JSON
#    include "corecel/io/OutputInterfaceAdapter.hh"
#    include "corecel/sys/EnvironmentIO.json.hh"
#    include "corecel/sys/MemRegistryIO.json.hh"

#    include "RunInputIO.json.hh"
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared Celeritas params on the master thread.
 *
 * The shared params will be uninitialized if Celeritas offloading is disabled.
 * In this case, a new output registry will be created for the Geant4
 * diagnostics. If Celeritas offloading is enabled, diagnostics will be added
 * to the output registry in the \c SharedParams.
 */
GeantDiagnostics::GeantDiagnostics(SharedParams const& params)
{
    CELER_EXPECT(static_cast<bool>(params)
                 == !SharedParams::CeleritasDisabled());

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    // Get (lazily creating)
    SPOutputRegistry output_reg = params.output_reg();
    size_type num_threads = params.num_streams();

    // Create the timer output and add to output registry
    timer_output_ = std::make_shared<TimerOutput>(num_threads);
    output_reg->insert(timer_output_);

    auto& global_setup = *GlobalSetup::Instance();
    if (global_setup.StepDiagnostic())
    {
        // Create the track step diagnostic and add to output registry
        auto num_bins = GlobalSetup::Instance()->GetStepDiagnosticBins();
        step_diagnostic_
            = std::make_shared<GeantStepDiagnostic>(num_bins, num_threads);
        output_reg->insert(step_diagnostic_);

        // Add the Celeritas step diagnostic if Celeritas offloading is enabled
        if (params)
        {
            auto step_diagnostic = std::make_shared<celeritas::StepDiagnostic>(
                params.Params()->action_reg()->next_id(),
                params.Params()->particle(),
                num_bins,
                num_threads);
            params.Params()->action_reg()->insert(step_diagnostic);
            output_reg->insert(step_diagnostic);
        }
    }

    if (!params)
    {
        // Celeritas core params didn't add system metadata: do it ourselves
#if CELERITAS_USE_JSON
        // Save system diagnostic information
        output_reg->insert(OutputInterfaceAdapter<MemRegistry>::from_const_ref(
            OutputInterface::Category::system,
            "memory",
            celeritas::mem_registry()));
        output_reg->insert(OutputInterfaceAdapter<Environment>::from_const_ref(
            OutputInterface::Category::system,
            "environ",
            celeritas::environment()));
#endif
        output_reg->insert(std::make_shared<BuildOutput>());

        // Save filename from global options (TODO: remove this hack)
        const_cast<SharedParams&>(params).set_output_filename(
            global_setup.setup_options().output_file);
    }

#if CELERITAS_USE_JSON
    // Save input options
    output_reg->insert(OutputInterfaceAdapter<RunInput>::from_const_ref(
        OutputInterface::Category::input, "*", global_setup.input()));
#endif

    // Create shared exception handler
    meh_ = std::make_shared<MultiExceptionHandler>();

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Clear diagnostics at the end of the run.
 *
 * This must be executed exactly *once* across all threads and at the end of
 * the run.
 */
void GeantDiagnostics::Finalize()
{
    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting diagnostics";
    if (meh_)
    {
        log_and_rethrow(std::move(*meh_));
    }
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
