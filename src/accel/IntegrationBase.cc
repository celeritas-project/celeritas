//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.cc
//---------------------------------------------------------------------------//
#include "IntegrationBase.hh"

#include <G4Threading.hh>

#include "corecel/Assert.hh"

#include "ExceptionConverter.hh"
#include "TimeOutput.hh"

#include "detail/IntegrationSingleton.hh"

using celeritas::detail::IntegrationSingleton;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set options before starting the run.
 *
 * This captures the input to indicate that options cannot be modified after
 * this point.
 */
void IntegrationBase::SetOptions(SetupOptions&& opts)
{
    IntegrationSingleton::instance().setup_options(std::move(opts));
}

//---------------------------------------------------------------------------//
/*!
 * Access whether Celeritas is set up, enabled, or uninitialized.
 *
 * This is only legal to call after \c SetOptions.
 */
OffloadMode IntegrationBase::GetMode() const
{
    return IntegrationSingleton::instance().mode();
}

//---------------------------------------------------------------------------//
/*!
 * End the run.
 */
void IntegrationBase::EndOfRunAction(G4Run const*)
{
    CELER_LOG(status) << "Finalizing Celeritas";

    auto& singleton = IntegrationSingleton::instance();

    // Record the run time
    auto time = singleton.stop_timer();

    // Remove local transporter
    singleton.finalize_local_transporter();

    if (G4Threading::IsMasterThread())
    {
        singleton.shared_params().timer()->RecordTotalTime(time);
        singleton.finalize_shared_params();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Access Celeritas shared params.
 */
CoreParams const& IntegrationBase::GetParams()
{
    auto& singleton = IntegrationSingleton::instance();
    CELER_TRY_HANDLE(
        {
            auto mode = singleton.shared_params().mode();
            CELER_VALIDATE(
                mode != OffloadMode::disabled
                    && mode != OffloadMode::uninitialized,
                << R"(cannot access shared params when Celeritas is disabled or outside of a run)");
        },
        ExceptionConverter{"celer.get.params"});
    return *singleton.shared_params().Params();
}

//---------------------------------------------------------------------------//
/*!
 * Access THREAD-LOCAL Celeritas core state data for user diagnostics.
 *
 * - This can \em only be called when Celeritas is enabled (not kill-offload,
 *   not disabled)
 * - This cannot be called from the main thread of an MT application.
 */
CoreStateInterface& IntegrationBase::GetState()
{
    auto& singleton = IntegrationSingleton::instance();
    CELER_TRY_HANDLE(
        {
            CELER_VALIDATE(
                !G4Threading::IsMultithreadedApplication()
                    || G4Threading::IsWorkerThread(),
                << R"(cannot call Integration::GetState from master thread in a multithreaded application)");
            CELER_VALIDATE(
                singleton.shared_params().mode() == OffloadMode::enabled,
                << R"(cannot access local state unless Celeritas is enabled)");
        },
        ExceptionConverter{"celer.get.state"});

    return singleton.local_transporter().GetState();
}

//---------------------------------------------------------------------------//
/*!
 * Initialize logging on first access.
 *
 * Since this is done during static initialization, it is guaranteed to be
 * thread safe. By ensuring \c IntegrationSingleton has been constructed, it
 * guarantees MPI is initialized (or disabled) before the first log message.
 */
IntegrationBase::IntegrationBase()
{
    IntegrationSingleton::instance().initialize_logger();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
