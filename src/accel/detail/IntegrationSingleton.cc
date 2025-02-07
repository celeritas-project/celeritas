//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.cc
//---------------------------------------------------------------------------//
#include "IntegrationSingleton.hh"

#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Static GLOBAL shared data.
 */
IntegrationSingleton& IntegrationSingleton::instance()
{
    static IntegrationSingleton is;
    return is;
}

//---------------------------------------------------------------------------//
/*!
 * Static THREAD-LOCAL Celeritas state data.
 */
LocalTransporter& IntegrationSingleton::local_transporter()
{
    static G4ThreadLocal LocalTransporter lt;
    return lt;
}

//---------------------------------------------------------------------------//
/*!
 * Static global setup options before constructing params.
 */
SetupOptions& IntegrationSingleton::setup_options()
{
    CELER_VALIDATE(
        !params_,
        << R"(options cannot be modified after Celeritas is constructed)");
    return options_;
}

//---------------------------------------------------------------------------//
/*!
 * Set up logging.
 */
void IntegrationSingleton::initialize_logger()
{
    auto initialize_impl = [this] {
        auto* run_man = G4RunManager::GetRunManager();
        CELER_VALIDATE(run_man,
                       << "logger cannot be set up before run manager");
        CELER_VALIDATE(!params_,
                       << "logger cannot be set up after shared params");
        celeritas::self_logger() = celeritas::MakeMTLogger(*run_man);
    };

    CELER_TRY_HANDLE(initialize_impl(),
                     ExceptionConverter{"celer.init.logger"});
}

//---------------------------------------------------------------------------//
/*!
 * Construct shared params on master (or single) thread.
 *
 * \todo The query for CeleritasDisabled initializes the environment before
 * we've had a chance to load the user setup options. Make sure we can update
 * the environment *first* when refactoring the setup.
 *
 * \note In Geant4 threading, \em only MT mode on non-master thread has
 *   \c G4Threading::IsWorkerThread()==true. For MT mode, the master thread
 *   does not track any particles. For single-thread mode, the master thread
 *   \em does do work.
 */
void IntegrationSingleton::initialize_shared_params()
{
    ExceptionConverter call_g4exception{"celer.init.global"};

    if (G4Threading::IsMasterThread())
    {
        CELER_LOG_LOCAL(debug) << "Initializing shared params";

        auto initialize_impl = [this] {
            CELER_VALIDATE(
                !params_,
                << R"(BeginOfRunAction cannot be called more than once)");
            params_.Initialize(options_);
        };

        CELER_TRY_HANDLE(initialize_impl(), call_g4exception);
    }
    else
    {
        CELER_LOG_LOCAL(debug) << "Initializing worker";
        CELER_ASSERT(G4Threading::IsMultithreadedApplication());

        auto initialize_impl = [this] {
            CELER_VALIDATE(
                params_,
                << R"(BeginOfRunAction was not called on master thread)");
            params_.InitializeWorker(options_);
        };

        CELER_TRY_HANDLE(initialize_impl(), call_g4exception);
    }

    CELER_ENSURE(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local transporter.
 *
 * Note that this uses the thread-local static data. It *must not* be called
 * from the master thread in a multithreaded run.
 *
 * \return Whether Celeritas offloading is enabled
 */
bool IntegrationSingleton::initialize_local_transporter()
{
    CELER_EXPECT(params_);

    if (G4Threading::IsMultithreadedApplication()
        && G4Threading::IsMasterThread())
    {
        // Cannot construct local transporter on master MT thread
        return false;
    }

    CELER_EXPECT(!G4Threading::IsMultithreadedApplication()
                 || G4Threading::IsWorkerThread());

    if (params_.mode() == celeritas::SharedParams::Mode::disabled)
    {
        CELER_LOG_LOCAL(debug)
            << R"(Skipping state construction since Celeritas is completely disabled)";
        return false;
    }

    if (params_.mode() == celeritas::SharedParams::Mode::kill_offload)
    {
        // When "kill offload", we still need to intercept tracks
        CELER_LOG_LOCAL(debug)
            << R"(Skipping state construction with offload enabled: offload-compatible tracks will be killed immediately)";
        return true;
    }

    CELER_LOG_LOCAL(debug) << "Constructing local state";

    auto initialize_impl = [this] {
        auto& lt = IntegrationSingleton::local_transporter();
        CELER_VALIDATE(!lt,
                       << "local thread " << G4Threading::G4GetThreadId() + 1
                       << " cannot be initialized more than once");
        lt.Initialize(options_, params_);
    };
    CELER_TRY_HANDLE(initialize_impl(), ExceptionConverter{"celer.init.local"});
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Destroy local transporter.
 */
void IntegrationSingleton::finalize_local_transporter()
{
    CELER_EXPECT(params_);

    if (G4Threading::IsMultithreadedApplication()
        && G4Threading::IsMasterThread())
    {
        // Cannot destroy local transporter on master MT thread
        return;
    }

    if (params_.mode() != celeritas::SharedParams::Mode::enabled)
    {
        return;
    }

    CELER_LOG_LOCAL(debug) << "Destroying local state";

    auto finalize_impl = [] {
        auto& lt = IntegrationSingleton::local_transporter();
        CELER_VALIDATE(lt,
                       << "local thread " << G4Threading::G4GetThreadId() + 1
                       << " cannot be finalized more than once");
        lt.Finalize();
    };

    CELER_TRY_HANDLE(finalize_impl(),
                     ExceptionConverter{"celer.finalize.local"});
}

//---------------------------------------------------------------------------//
/*!
 * Destroy params.
 */
void IntegrationSingleton::finalize_shared_params()
{
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

    auto finalize_impl = [this] {
        CELER_VALIDATE(params_, << "params cannot be finalized more than once");
        params_.Finalize();
    };

    CELER_TRY_HANDLE(finalize_impl(),
                     ExceptionConverter{"celer.finalize.global"});
}

//---------------------------------------------------------------------------//
/*!
 * Construct and set up options messenger.
 */
IntegrationSingleton::IntegrationSingleton() = default;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
