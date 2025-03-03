//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.cc
//---------------------------------------------------------------------------//
#include "IntegrationBase.hh"

#include <G4Threading.hh>

#include "corecel/Assert.hh"

#include "detail/IntegrationSingleton.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Edit options before starting the run.
 */
void IntegrationBase::SetOptions(SetupOptions&& opts)
{
    detail::IntegrationSingleton::instance().setup_options(std::move(opts));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on non-worker thread in MT mode.
 */
void IntegrationBase::BuildForMaster()
{
    CELER_VALIDATE(
        G4Threading::IsMasterThread()
            && G4Threading::IsMultithreadedApplication(),
        << R"(BuildForMaster called from a worker thread or non-MT code)");

    detail::IntegrationSingleton::instance().initialize_logger();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize during ActionInitialization on a worker thread or serial mode.
 *
 * We guard against \c Build being called from \c BuildForMaster since we might
 * add worker-specific code here.
 */
void IntegrationBase::Build()
{
    if (G4Threading::IsMasterThread())
    {
        CELER_VALIDATE(!G4Threading::IsMultithreadedApplication(),
                       << "cannot call Integration::Build from worker thread "
                          "in a multithreaded application");

        detail::IntegrationSingleton::instance().initialize_logger();
    }
}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void IntegrationBase::EndOfRunAction(G4Run const*)
{
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

    auto& singleton = detail::IntegrationSingleton::instance();

    // Remove local transporter
    singleton.finalize_local_transporter();

    if (G4Threading::IsMasterThread())
    {
        singleton.finalize_shared_params();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
