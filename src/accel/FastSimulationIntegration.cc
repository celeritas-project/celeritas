//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.cc
//---------------------------------------------------------------------------//
#include "FastSimulationIntegration.hh"

#include <G4Threading.hh>

#include "corecel/sys/Stopwatch.hh"

#include "ExceptionConverter.hh"
#include "TimeOutput.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Check fast simulation is set up.
 *
 * \todo This is currently a null op, but we should loop through the regions to
 * ensure that at least one of them is a celeritas::FastSimulationModel.
 */
void verify_fast_sim() {}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Access the public-facing integration singleton.
 */
FastSimulationIntegration& FastSimulationIntegration::Instance()
{
    static FastSimulationIntegration tmi;
    return tmi;
}

//---------------------------------------------------------------------------//
/*!
 * Start the run, initializing Celeritas options.
 */
void FastSimulationIntegration::BeginOfRunAction(G4Run const*)
{
    Stopwatch get_setup_time;

    auto& singleton = detail::IntegrationSingleton::instance();

    if (G4Threading::IsMasterThread())
    {
        singleton.initialize_shared_params();
    }

    bool enable_offload = singleton.initialize_local_transporter();

    if (enable_offload)
    {
        // Set tracking manager on workers when Celeritas is not fully disabled
        CELER_LOG(debug) << "Verifying fast simulation";

        CELER_TRY_HANDLE(verify_fast_sim(),
                         ExceptionConverter{"celer.init.verify"});
    }

    if (G4Threading::IsMasterThread())
    {
        singleton.shared_params().timer()->RecordSetupTime(get_setup_time());
        singleton.start_timer();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
FastSimulationIntegration::FastSimulationIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
