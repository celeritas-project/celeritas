//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.cc
//---------------------------------------------------------------------------//
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "corecel/io/Logger.hh"

#include "IntegrationSingleton.hh"

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

    CELER_TRY_HANDLE(initialize_impl, ExceptionConverter{"celer.init.logger"});
}

//---------------------------------------------------------------------------//
/*!
 * Construct shared params on master (or single) thread.
 */
void IntegrationSingleton::initialize_shared_params()
{
    ExceptionConverter call_g4exception{"celer.init.global"};

    if (!G4Threading::IsMultithreadedApplication()
        || G4Threading::IsMasterThread())
    {
        CELER_LOG_LOCAL(debug) << "Initializing shared params";
        CELER_TRY_HANDLE(params_->Initialize(*options_), call_g4exception);
    }
    else
    {
        CELER_LOG_LOCAL(debug) << "Initializing worker";
        CELER_TRY_HANDLE(SharedParams::InitializeWorker(*options_),
                         call_g4exception);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local transporter.
 *
 * Note that this uses the thread-local static data. It *must not* be called
 * from the master thread in a multithreaded run.
 */
void IntegrationSingleton::initialize_local_transporter()
{
    CELER_EXPECT(!G4Threading::IsMultithreadedApplication()
                 || !G4Threading::IsMasterThread());
    CELER_LOG_LOCAL(debug) << "Constructing local state";

    auto& lt = IntegrationSingleton::local_transporter();
    CELER_VALIDATE(!lt,
                   << "local thread " << G4Threading::G4GetThreadId() + 1
                   << " cannot be initialized more than once")
    CELER_TRY_HANDLE(lt.Initialize(options_, params_),
                     ExceptionConverter{"celer.init.local"});
}

//---------------------------------------------------------------------------//
/*!
 * Destroy local transporter.
 */
void IntegrationSingleton::finalize_local_transporter()
{
    CELER_EXPECT(!G4Threading::IsMultithreadedApplication()
                 || !G4Threading::IsMasterThread());

    CELER_LOG_LOCAL(debug) << "Destroying local state";

    auto& lt = IntegrationSingleton::local_transporter();
    CELER_VALIDATE(!lt,
                   << "local thread " << G4Threading::G4GetThreadId() + 1
                   << " cannot be initialized more than once")

    CELER_TRY_HANDLE(local_->Finalize(), call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Destroy params.
 */
void IntegrationSingleton::finalize_shared_params() {}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
