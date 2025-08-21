//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/IntegrationSingleton.cc
//---------------------------------------------------------------------------//
#include "IntegrationSingleton.hh"

#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4MuonMinus.hh>
#include <G4MuonPlus.hh>
#include <G4ParticleDefinition.hh>
#include <G4Positron.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#include "../ExceptionConverter.hh"
#include "../Logger.hh"
#include "../SetupOptionsMessenger.hh"
#include "../TimeOutput.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Verify that all particles in \c SetupOptions::offload_particles user-defined
 * list are supported by Celeritas.
 */
void validate_offloaded_particles(SetupOptions::VecG4PD const& user)
{
    if (user.empty())
    {
        // Celeritas will use default hardcoded list; nothing to do
        return;
    }

    auto const supported = IntegrationSingleton::supported_offload_particles();
    auto find = [&supported](G4ParticleDefinition* user) -> bool {
        return std::any_of(
            supported.begin(),
            supported.end(),
            [&user](G4ParticleDefinition* p) {
                return (p->GetPDGEncoding() == user->GetPDGEncoding());
            });
    };

    for (auto const& pd : user)
    {
        CELER_ASSERT(pd);
        CELER_VALIDATE(find(pd),
                       << "Particle \"" << pd->GetParticleName()
                       << "\" (PDG = " << pd->GetPDGEncoding()
                       << ") is not available in Celeritas");
    }
}
//---------------------------------------------------------------------------//
};  // namespace

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
 * Get a list of all supported particles.
 */
IntegrationSingleton::VecG4PD
IntegrationSingleton::supported_offload_particles()
{
    static G4ParticleDefinition* const supported_particles[] = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
        G4MuonMinus::Definition(),
        G4MuonPlus::Definition(),
    };

    return {std::begin(supported_particles), std::end(supported_particles)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the list of default particles offloaded in Geant4 applications.
 *
 * If no user-defined list is provided, this defaults to simulating EM showers.
 */
IntegrationSingleton::VecG4PD IntegrationSingleton::default_offload_particles()
{
    static G4ParticleDefinition* const default_particles[] = {
        G4Electron::Definition(),
        G4Positron::Definition(),
        G4Gamma::Definition(),
    };

    return {std::begin(default_particles), std::end(default_particles)};
}

//---------------------------------------------------------------------------//
/*!
 * Assign global setup options before constructing params.
 */
void IntegrationSingleton::setup_options(SetupOptions&& opts)
{
    CELER_TRY_HANDLE(
        {
            CELER_VALIDATE(
                !params_,
                << R"(options cannot be set after Celeritas is constructed)");
            validate_offloaded_particles(opts.offload_particles);
            options_ = std::move(opts);
        },
        ExceptionConverter{"celer.setup"});
    if (!options_)
    {
        CELER_LOG(warning)
            << R"(SetOptions called with incomplete input: you must use the UI to update before /run/initialize)";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Set up logging.
 */
void IntegrationSingleton::initialize_logger()
{
    CELER_TRY_HANDLE(
        {
            auto* run_man = G4RunManager::GetRunManager();
            CELER_VALIDATE(run_man,
                           << "logger cannot be set up before run manager");
            CELER_VALIDATE(!params_,
                           << "logger cannot be set up after shared params");
            celeritas::world_logger() = celeritas::MakeMTWorldLogger(*run_man);
            celeritas::self_logger() = celeritas::MakeMTSelfLogger(*run_man);
        },
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
        CELER_LOG(debug) << "Initializing shared params";
        CELER_TRY_HANDLE(
            {
                CELER_VALIDATE(
                    options_,
                    << R"(SetOptions or UI entries were not completely set before BeginRun)");
                CELER_VALIDATE(
                    !params_,
                    << R"(BeginOfRunAction cannot be called more than once)");
                params_.Initialize(options_);
            },
            call_g4exception);
    }
    else
    {
        CELER_LOG(debug) << "Initializing worker";
        CELER_TRY_HANDLE(
            {
                CELER_ASSERT(G4Threading::IsMultithreadedApplication());
                CELER_VALIDATE(
                    params_,
                    << R"(BeginOfRunAction was not called on master thread)");
                params_.InitializeWorker(options_);
            },
            call_g4exception);
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

    if (params_.mode() == celeritas::SharedParams::Mode::disabled)
    {
        CELER_LOG(debug)
            << R"(Skipping state construction since Celeritas is completely disabled)";
        return false;
    }

    if (G4Threading::IsMultithreadedApplication()
        && G4Threading::IsMasterThread())
    {
        // Cannot construct local transporter on master MT thread
        return false;
    }

    CELER_ASSERT(!G4Threading::IsMultithreadedApplication()
                 || G4Threading::IsWorkerThread());

    if (params_.mode() == celeritas::SharedParams::Mode::kill_offload)
    {
        // When "kill offload", we still need to intercept tracks
        CELER_LOG(debug)
            << R"(Skipping state construction with offload enabled: offload-compatible tracks will be killed immediately)";
        return true;
    }

    CELER_LOG(debug) << "Constructing local state";

    CELER_TRY_HANDLE(
        {
            auto& lt = IntegrationSingleton::local_transporter();
            CELER_VALIDATE(!lt,
                           << "local thread "
                           << G4Threading::G4GetThreadId() + 1
                           << " cannot be initialized more than once");
            lt.Initialize(options_, params_);
        },
        ExceptionConverter("celer.init.local"));
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Destroy local transporter.
 */
void IntegrationSingleton::finalize_local_transporter()
{
    CELER_EXPECT(params_);

    if (params_.mode() != celeritas::SharedParams::Mode::enabled)
    {
        return;
    }

    if (G4Threading::IsMultithreadedApplication()
        && G4Threading::IsMasterThread())
    {
        // Cannot destroy local transporter on master MT thread
        return;
    }

    CELER_LOG(debug) << "Destroying local state";

    CELER_TRY_HANDLE(
        {
            auto& lt = IntegrationSingleton::local_transporter();
            CELER_VALIDATE(lt,
                           << "local thread "
                           << G4Threading::G4GetThreadId() + 1
                           << " cannot be finalized more than once");
            params_.timer()->RecordActionTime(lt.GetActionTime());
            lt.Finalize();
        },
        ExceptionConverter("celer.finalize.local"));
}

//---------------------------------------------------------------------------//
/*!
 * Destroy params.
 */
void IntegrationSingleton::finalize_shared_params()
{
    CELER_LOG(status) << "Finalizing Celeritas";
    CELER_TRY_HANDLE(
        {
            CELER_VALIDATE(params_,
                           << "params cannot be finalized more than once");
            params_.Finalize();
        },
        ExceptionConverter("celer.finalize.global"));
}

//---------------------------------------------------------------------------//
/*!
 * Construct and set up the singleton.
 *
 * Using unique pointers for MPI and messenger allow us to catch errors they
 * may throw during construction.
 */
IntegrationSingleton::IntegrationSingleton()
{
    CELER_TRY_HANDLE(
        {
            scoped_mpi_ = std::make_unique<ScopedMpiInit>();
            messenger_ = std::make_unique<SetupOptionsMessenger>(&options_);
        },
        ExceptionConverter{"celer.init.singleton"});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
