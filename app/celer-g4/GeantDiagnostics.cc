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
 * TODO
 */
GeantDiagnostics::GeantDiagnostics(SPConstParams params) : params_(params)
{
    CELER_EXPECT(params_);

    // No diagnostics enabled
    if (!this->any_enabled())
        return;

    CELER_LOG_LOCAL(status) << "Initializing Geant4 diagnostics";

    output_reg_ = *params_ ? params_->Params()->output_reg()
                           : std::make_shared<OutputRegistry>();

    output_filename_ = GlobalSetup::Instance()->GetSetupOptions()->output_file;
    if (output_filename_.empty())
    {
        output_filename_ = "celer-g4.out.json";
    }

    if (GlobalSetup::Instance()->StepDiagnostic())
    {
        // Create the track step diagnostic
        size_type num_threads = [this] {
            if (*params_)
            {
                return params_->Params()->max_streams();
            }
            auto* run_man = G4RunManager::GetRunManager();
            CELER_ASSERT(run_man);
            return size_type(get_num_threads(*run_man));
        }();
        step_diagnostic_ = std::make_shared<GeantStepDiagnostic>(
            GlobalSetup::Instance()->GetStepDiagnosticBins(), num_threads);

        // Add to output interface
        output_reg_->insert(step_diagnostic_);
    }

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Write out diagnostics if offloading is disabled.
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
    if (*params_)
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
