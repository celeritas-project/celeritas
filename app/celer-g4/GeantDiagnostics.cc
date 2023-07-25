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

    // No diagnostics enabled
    if (!this->any_enabled())
        return;

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    output_filename_ = GlobalSetup::Instance()->GetSetupOptions()->output_file;
    if (output_filename_.empty())
    {
        output_filename_ = "celer-g4.out.json";
    }

    SPOutputRegistry output_reg = *params ? params->Params()->output_reg()
                                          : std::make_shared<OutputRegistry>();

    if (GlobalSetup::Instance()->StepDiagnostic())
    {
        // Create the track step diagnostic
        size_type num_threads = [&params] {
            if (*params)
            {
                return params->Params()->max_streams();
            }
            auto* run_man = G4RunManager::GetRunManager();
            CELER_ASSERT(run_man);
            return size_type(get_num_threads(*run_man));
        }();
        step_diagnostic_ = std::make_shared<GeantStepDiagnostic>(
            GlobalSetup::Instance()->GetStepDiagnosticBins(), num_threads);

        // Add to output registry
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

    // No diagnostics enabled
    if (!this->any_enabled())
        return;

    // Output was already written
    if (!output_reg_)
        return;

#if CELERITAS_USE_JSON
    CELER_LOG(info) << "Writing Geant4 diagnostic output to \""
                    << output_filename_ << '"';

    std::ofstream outf(output_filename_);
    CELER_VALIDATE(
        outf, << "failed to open output file at \"" << output_filename_ << '"');
    output_reg_->output(&outf);
#else
    CELER_LOG(warning)
        << "JSON support is not enabled, so no output will be written to \""
        << output_filename_ << '"';
#endif
}

//---------------------------------------------------------------------------//
/*!
 * True if any Geant4 diagnostics are enabled.
 */
bool GeantDiagnostics::any_enabled() const
{
    return GlobalSetup::Instance()->StepDiagnostic();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
