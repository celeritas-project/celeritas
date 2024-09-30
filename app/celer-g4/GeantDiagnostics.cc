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
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/MemRegistryIO.json.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/user/StepDiagnostic.hh"

#include "GlobalSetup.hh"
#include "RunInputIO.json.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Add outputs to a queue *only from the main thread*.
 *
 * This is not thread-safe.
 */
void GeantDiagnostics::register_output(VecOutputInterface&& output)
{
    CELER_LOG_LOCAL(debug) << "Registering " << output.size()
                           << " output interfaces";

    auto& q = queued_output();
    if (q.empty())
    {
        q = std::move(output);
    }
    else
    {
        q.insert(q.end(),
                 std::make_move_iterator(output.begin()),
                 std::make_move_iterator(output.end()));
        output.clear();
    }
}

//---------------------------------------------------------------------------//
auto GeantDiagnostics::queued_output() -> VecOutputInterface&
{
    static VecOutputInterface output;
    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from shared Celeritas params on the master thread.
 *
 * The shared params will be uninitialized if Celeritas offloading is disabled.
 * In this case, a new output registry will be created for the Geant4
 * diagnostics.
 */
GeantDiagnostics::GeantDiagnostics(SharedParams const& params)
{
    CELER_EXPECT(static_cast<bool>(params)
                 == !SharedParams::CeleritasDisabled());

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    // Get output registry
    auto output_reg = params.output_reg();
    CELER_ASSERT(output_reg);
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
            StepDiagnostic::make_and_insert(*params.Params(), num_bins);
        }
    }

    if (!params)
    {
        // Celeritas core params didn't add system metadata: do it ourselves
        // to save system diagnostic information
        output_reg->insert(OutputInterfaceAdapter<MemRegistry>::from_const_ref(
            OutputInterface::Category::system,
            "memory",
            celeritas::mem_registry()));
        output_reg->insert(OutputInterfaceAdapter<Environment>::from_const_ref(
            OutputInterface::Category::system,
            "environ",
            celeritas::environment()));
        output_reg->insert(std::make_shared<BuildOutput>());
    }

    // Save detectors
    for (auto&& output : queued_output())
    {
        output_reg->insert(std::move(output));
    }
    queued_output().clear();

    // Save input options
    output_reg->insert(OutputInterfaceAdapter<RunInput>::from_const_ref(
        OutputInterface::Category::input, "*", global_setup.input()));

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
    if (CELER_UNLIKELY(!queued_output().empty()))
    {
        CELER_LOG(warning) << "Output interfaces were added after the run "
                              "began: output will be missing";
    }

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
