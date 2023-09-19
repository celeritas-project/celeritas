//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantDiagnostics.cc
//---------------------------------------------------------------------------//
#include "GeantDiagnostics.hh"

#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/global/CoreParams.hh"

#include "G4RunManager.hh"
#include "GlobalSetup.hh"

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
GeantDiagnostics::GeantDiagnostics(SPConstParams params)
{
    CELER_EXPECT(params);
    CELER_EXPECT(*params || !celeritas::getenv("CELER_DISABLE").empty());

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    SPOutputRegistry output_reg = *params ? params->Params()->output_reg()
                                          : std::make_shared<OutputRegistry>();

    size_type num_threads = [&params] {
        if (*params)
        {
            return params->Params()->max_streams();
        }
        auto* run_man = G4RunManager::GetRunManager();
        CELER_ASSERT(run_man);
        return size_type(get_num_threads(*run_man));
    }();

    // Create the timer output and add to output registry
    timer_output_ = std::make_shared<TimerOutput>(num_threads);
    output_reg->insert(timer_output_);

    if (GlobalSetup::Instance()->StepDiagnostic())
    {
        // Create the track step diagnostic and add to output registry
        step_diagnostic_ = std::make_shared<GeantStepDiagnostic>(
            GlobalSetup::Instance()->GetStepDiagnosticBins(), num_threads);
        output_reg->insert(step_diagnostic_);
    }

    if (!*params)
    {
        output_reg_ = std::move(output_reg);
    }

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Write out diagnostics if Celeritas has not been initialized.
 *
 * This must be executed exactly *once* across all threads and at the end of
 * the run.
 */
void GeantDiagnostics::Finalize()
{
    CELER_EXPECT(*this);

    // Output written by \c SharedParams
    if (!output_reg_)
        return;

    auto filename = GlobalSetup::Instance()->GetSetupOptions()->output_file;
#if CELERITAS_USE_JSON
    CELER_LOG(info) << "Writing Geant4 diagnostic output to \"" << filename
                    << '"';

    std::ofstream outf(filename);
    CELER_VALIDATE(outf,
                   << "failed to open output file at \"" << filename << '"');
    output_reg_->output(&outf);
#else
    CELER_LOG(warning)
        << "JSON support is not enabled, so no output will be written to \""
        << filename << '"';
#endif
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
