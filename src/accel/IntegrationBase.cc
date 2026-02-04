//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.cc
//---------------------------------------------------------------------------//
#include "IntegrationBase.hh"

#include <G4Threading.hh>

#include "corecel/Assert.hh"
#include "accel/LocalTransporter.hh"

#include "ExceptionConverter.hh"

#include "detail/IntegrationSingleton.hh"

using celeritas::detail::IntegrationSingleton;

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
//! \name User integration points

/*!
 * Set options before starting the run.
 *
 * This captures the input to indicate that options cannot be modified by the
 * framework after this point.
 */
void IntegrationBase::SetOptions(SetupOptions&& opts)
{
    IntegrationSingleton::instance().setup_options(std::move(opts));
}

/*!
 * Start the run.
 *
 * This handles shared/local setup and calls verify_setup if offload is
 * enabled.
 */
void IntegrationBase::BeginOfRunAction(G4Run const*)
{
    auto& singleton = IntegrationSingleton::instance();

    // Initialize shared params and local transporter
    bool enable_offload = singleton.initialize_offload();

    if (enable_offload)
    {
        // Allow derived classes to perform their specific verification
        CELER_TRY_HANDLE(this->verify_local_setup(),
                         ExceptionConverter{"celer.init.verify"});
    }
}

/*!
 * End the run.
 */
void IntegrationBase::EndOfRunAction(G4Run const*)
{
    auto& singleton = IntegrationSingleton::instance();

    singleton.finalize_offload();
}

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Low-level Celeritas accessors

/*!
 * Access whether Celeritas is set up, enabled, or uninitialized.
 *
 * This is only legal to call after \c SetOptions.
 */
OffloadMode IntegrationBase::GetMode() const
{
    return IntegrationSingleton::instance().mode();
}

/*!
 * Access \em global Celeritas shared params during a run, if not disabled.
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

/*!
 * Access \em thread-local Celeritas core state data for user diagnostics.
 *
 * - This can \em only be called when Celeritas is enabled (not kill-offload,
 *   not disabled).
 * - This cannot be called from the main thread of an MT application.
 */
CoreStateInterface& IntegrationBase::GetState()
{
    auto& singleton = IntegrationSingleton::instance();
    LocalTransporter* lt{nullptr};
    CELER_TRY_HANDLE(
        {
            CELER_VALIDATE(
                !G4Threading::IsMultithreadedApplication()
                    || G4Threading::IsWorkerThread(),
                << R"(cannot call Integration::GetState from master thread in a multithreaded application)");
            CELER_VALIDATE(
                singleton.shared_params().mode() == OffloadMode::enabled,
                << R"(cannot access local state unless Celeritas is enabled)");
            lt = dynamic_cast<LocalTransporter*>(&singleton.local_offload());
            CELER_VALIDATE(
                lt, << R"(cannot access EM state when not using EM offload)");
        },
        ExceptionConverter{"celer.get.state"});

    return lt->GetState();
}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Initialize MPI and logging on first access.
 */
IntegrationBase::IntegrationBase()
{
    IntegrationSingleton::instance();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
