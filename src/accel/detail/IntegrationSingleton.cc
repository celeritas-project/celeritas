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
#include "corecel/sys/ScopedMpiInit.hh"
#include "geocel/GeantUtils.hh"

#include "LoggerImpl.hh"
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
 * list are valid and supported by Celeritas when non-empty. Return user or
 * default list accordingly.
 */
SetupOptions::VecG4PD
validate_and_return_offloaded(std::optional<SetupOptions::VecG4PD> const& user)
{
    if (!user)
    {
        // Celeritas will use default hardcoded list; nothing to do
        return SharedParams::default_offload_particles();
    }

    auto const supported = SharedParams::supported_offload_particles();
    auto find = [&supported](G4ParticleDefinition* user) -> bool {
        return std::any_of(
            supported.begin(),
            supported.end(),
            [&user](G4ParticleDefinition* p) {
                return (p->GetPDGEncoding() == user->GetPDGEncoding());
            });
    };

    for (auto const& pd : *user)
    {
        CELER_ASSERT(pd);
        CELER_VALIDATE(find(pd),
                       << "Particle " << StreamablePD{pd}
                       << " is not available in Celeritas");
    }
    return *user;
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
    auto& offload = IntegrationSingleton::local_offload_ptr();
    if (!offload)
    {
        offload = std::make_unique<LocalTransporter>();
    }
    auto* lt = dynamic_cast<LocalTransporter*>(offload.get());
    CELER_VALIDATE(lt,
                   << "Cannot access LocalTransporter when "
                      "LocalOpticalOffload is being used");
    return *lt;
}

//---------------------------------------------------------------------------//
/*!
 * Static THREAD-LOCAL Celeritas optical state data.
 */
LocalOpticalOffload& IntegrationSingleton::local_optical_offload()
{
    auto& offload = IntegrationSingleton::local_offload_ptr();
    if (!offload)
    {
        offload = std::make_unique<LocalOpticalOffload>();
    }
    auto* lt = dynamic_cast<LocalOpticalOffload*>(offload.get());
    CELER_VALIDATE(lt,
                   << "Cannot access LocalOpticalOffload when "
                      "LocalTransporter is being used");
    return *lt;
}

//---------------------------------------------------------------------------//
/*!
 * Access the thread-local offload interface.
 */
LocalOffloadInterface& IntegrationSingleton::local_offload()
{
    if (this->optical_offload())
    {
        return IntegrationSingleton::local_optical_offload();
    }
    return IntegrationSingleton::local_transporter();
}

//---------------------------------------------------------------------------//
/*!
 * Assign global setup options after run manager initialization but before run.
 */
void IntegrationSingleton::setup_options(SetupOptions&& opts)
{
    CELER_TRY_HANDLE(
        {
            // Run manager initialization requires no G4ParticleDef exists
            CELER_VALIDATE(
                G4RunManager::GetRunManager(),
                << R"(options cannot be set before G4RunManager is constructed)");
            // SharedParams require options to be set at BeginOfRun
            CELER_VALIDATE(
                !params_,
                << R"(options cannot be set after Celeritas is constructed)");
            offloaded_ = validate_and_return_offloaded(opts.offload_particles);
            options_ = std::move(opts);
        },
        ExceptionConverter{"celer.setup"});
    if (!options_)
    {
        CELER_LOG(warning)
            << R"(SetOptions called with incomplete input: you must use the UI to update before /run/initialize)";
    }

    CELER_ENSURE(!offloaded_.empty() || this->optical_offload());
}

//---------------------------------------------------------------------------//
/*!
 * Access whether Celeritas is set up, enabled, or uninitialized.
 */
OffloadMode IntegrationSingleton::mode() const
{
    if (offloaded_.empty())
    {
        CELER_LOG(warning) << "GetMode must not be called before SetOptions";
        return OffloadMode::uninitialized;
    }

    return SharedParams::GetMode();
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

                // Update logger in case run manager has changed number of
                // threads, or user called initialization after run manager
                this->update_logger();

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

    if (params_.mode() == OffloadMode::disabled)
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

    if (params_.mode() == OffloadMode::kill_offload)
    {
        // When "kill offload", we still need to intercept tracks
        CELER_LOG(debug)
            << R"(Skipping state construction with offload enabled: offload-compatible tracks will be killed immediately)";
        return true;
    }

    CELER_LOG(debug) << "Constructing local state";

    CELER_TRY_HANDLE(
        {
            auto& lt = this->local_offload();
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

    if (params_.mode() != OffloadMode::enabled)
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
            auto& lt = this->local_offload();
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
            this->update_logger();
        },
        ExceptionConverter{"celer.init.singleton"});
}

//---------------------------------------------------------------------------//
/*!
 * Static THREAD-LOCAL Celeritas offload.
 */
auto IntegrationSingleton::local_offload_ptr() -> UPOffload&
{
    static G4ThreadLocal UPOffload offload;
    return offload;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the local optical offload is used.
 */
bool IntegrationSingleton::optical_offload() const
{
    return options_.optical
           && std::holds_alternative<inp::OpticalOffloadGenerator>(
               options_.optical->generator);
}

//---------------------------------------------------------------------------//
/*!
 * Create or update the number of threads for the logger.
 */
void IntegrationSingleton::update_logger()
{
    if (auto* run_man = G4RunManager::GetRunManager())
    {
        if (!have_created_logger_)
        {
            celeritas::world_logger() = celeritas::MakeMTWorldLogger(*run_man);
            celeritas::self_logger() = celeritas::MakeMTSelfLogger(*run_man);
            have_created_logger_ = true;
        }
        else
        {
            if (auto* handle
                = celeritas::world_logger().handle().target<MtSelfWriter>())
            {
                // Update thread count
                *handle = MtSelfWriter{get_geant_num_threads(*run_man)};
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
