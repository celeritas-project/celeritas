//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GeantDiagnostics.cc
//---------------------------------------------------------------------------//
#include "GeantDiagnostics.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
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
GeantDiagnostics::GeantDiagnostics(SharedParams const& params)
{
    CELER_EXPECT(params || SharedParams::CeleritasDisabled());

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    // Get (lazily creating)
    SPOutputRegistry output_reg = params.output_reg();
    size_type num_threads = params.num_streams();

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

    if (!params)
    {
        // Write output on finalization
        output_reg_ = std::move(output_reg);
    }

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Write out diagnostics if Celeritas offloading is disabled.
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
    if (filename.empty() && !CELERITAS_USE_JSON)
    {
        filename = "celeritas.json";
        CELER_LOG(warning)
            << "No diagnostic output filename specified: using '" << filename
            << '"';
    }

    if (!CELERITAS_USE_JSON)
    {
        CELER_LOG(info) << "Writing Geant4 diagnostic output to \"" << filename
                        << '"';

        std::ofstream outf(filename);
        CELER_VALIDATE(
            outf, << "failed to open output file at \"" << filename << '"');
        output_reg_->output(&outf);
    }
    else
    {
        CELER_LOG(warning) << "JSON support is not enabled, so no output will "
                              "be written to \""
                           << filename << '"';
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
