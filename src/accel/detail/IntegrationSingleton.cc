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
 * list are valid and supported by Celeritas when non-empty.
 *
 * Return user or default list accordingly.
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
 * Access the thread-local offload interface.
 *
 * The first time this is called in an execution, we look at the options to
 * determine whether to create:
 *  - an EM track offload interface (LocalTransporter, which will send to the
 *    Celeritas EM core loop)
 *  - an optical track offload interface (TBD, which will send to a standalone
 *    optical loop)
 *  - an optical *generator* offload interface (LocalOpticalGenOffload, used
 *    for cherenkov/scintillation photons)
 */
LocalOffloadInterface& IntegrationSingleton::local_offload()
{
    static G4ThreadLocal UPOffload offload;

    if (CELER_UNLIKELY(!offload))
    {
        if (!options_)
        {
            // Cannot construct offload before options are set
            CELER_LOG_LOCAL(error)
                << R"(cannot access offload before options are set)";
        }
        if (options_.optical
            && std::holds_alternative<inp::OpticalOffloadGenerator>(
                options_.optical->generator))
        {
            offload = std::make_unique<LocalOpticalGenOffload>();
        }
        else
        {
            // TODO: if offloading direct optical tracks, return optical
            // offload
            offload = std::make_unique<LocalTransporter>();
        }
    }

    return *offload;
}

//---------------------------------------------------------------------------//
/*!
 * Access thread-local *track* offload interface (for anything that pushes a
 * track)
 */
TrackOffloadInterface& IntegrationSingleton::local_track_offload()
{
    auto* oi = dynamic_cast<TrackOffloadInterface*>(&this->local_offload());
    CELER_VALIDATE(oi,
                   << "Cannot access track offload when "
                      "LocalOpticalGenOffload is being used");
    return *oi;
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
 * Initialize shared params and thread-local transporter.
 *
 * This handles both global (master thread) and local (worker thread)
 * initialization. In MT mode, the master thread initializes shared params,
 * and worker threads initialize their local state.
 *
 * \return Whether Celeritas offloading is enabled
 */
bool IntegrationSingleton::initialize_offload()
{
    if (G4Threading::IsMasterThread())
    {
        failed_setup_ = false;

        ExceptionConverter call_g4exception{"celer.init.global"};
        CELER_TRY_HANDLE(this->initialize_master_impl(), call_g4exception);
        failed_setup_ = call_g4exception.forwarded();

        // Start the run timer
        get_time_ = {};
    }
    else if (!failed_setup_)
    {
        CELER_TRY_HANDLE(this->initialize_worker_impl(),
                         ExceptionConverter{"celer.init.worker"});
    }
    CELER_ASSERT(params_ || failed_setup_);

    // Now initialize local transporter
    if (CELER_UNLIKELY(failed_setup_))
    {
        CELER_LOG_LOCAL(debug)
            << R"(Skipping local initialization due to failure)";
        return false;
    }

    if (params_.mode() == OffloadMode::disabled)
    {
        CELER_LOG(debug)
            << R"(Skipping state construction since Celeritas is completely disabled)";
        return false;
    }

    if (G4Threading::IsMultithreadedApplication()
        && G4Threading::IsMasterThread())
    {
        // Do not construct local transporter on master MT thread
        return false;
    }

    if (params_.mode() == OffloadMode::kill_offload)
    {
        // When "kill offload", we still need to intercept tracks
        CELER_LOG(debug)
            << R"(Skipping state construction with offload enabled: offload-compatible tracks will be killed immediately)";
        return true;
    }

    CELER_TRY_HANDLE(this->initialize_local_impl(),
                     ExceptionConverter("celer.init.local"));
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Finalize thread-local transporter and (if main thread) shared params.
 */
void IntegrationSingleton::finalize_offload()
{
    if (CELER_UNLIKELY(failed_setup_))
    {
        return;
    }
    CELER_EXPECT(params_);

    // Finalize local transporter
    if (params_.mode() == OffloadMode::enabled)
    {
        if (!G4Threading::IsMultithreadedApplication()
            || !G4Threading::IsMasterThread())
        {
            CELER_TRY_HANDLE(this->finalize_local_impl(),
                             ExceptionConverter("celer.finalize.local"));
        }
    }

    // Finalize shared params on main thread
    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(this->finalize_shared_impl(),
                         ExceptionConverter("celer.finalize.global"));
    }
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
            CELER_LOG(debug) << "Celeritas logging redirected through Geant4";
        }
        else
        {
            if (celeritas::world_logger().handle().target<MtSelfWriter>())
            {
                // Update thread count by replacing log handle
                celeritas::world_logger().handle(
                    MtSelfWriter{get_geant_num_threads(*run_man)});
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Initialize shared params implementation.
 */
void IntegrationSingleton::initialize_master_impl()
{
    CELER_EXPECT(G4Threading::IsMasterThread());

    CELER_LOG(debug) << "Initializing shared params";
    CELER_VALIDATE(
        options_,
        << R"(SetOptions or UI entries were not completely set before BeginRun)");
    CELER_VALIDATE(!params_,
                   << R"(BeginOfRunAction cannot be called more than once)");

    Stopwatch get_setup_time;

    // Update logger in case run manager has changed number of
    // threads, or user called initialization after run manager
    this->update_logger();

    // Perform initialization
    params_.Initialize(options_);

    // Record the setup time after initialization
    params_.timer()->RecordSetupTime(get_setup_time());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize worker thread implementation.
 */
void IntegrationSingleton::initialize_worker_impl()
{
    CELER_EXPECT(G4Threading::IsMultithreadedApplication());

    CELER_LOG(debug) << "Initializing worker";
    CELER_VALIDATE(params_,
                   << R"(BeginOfRunAction was not called on master thread)");
    params_.InitializeWorker(options_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize local transporter implementation.
 */
void IntegrationSingleton::initialize_local_impl()
{
    CELER_EXPECT(!G4Threading::IsMultithreadedApplication()
                 || G4Threading::IsWorkerThread());

    CELER_LOG(debug) << "Constructing local state";
    auto& lt = this->local_offload();
    CELER_VALIDATE(!lt,
                   << "local thread " << G4Threading::G4GetThreadId() + 1
                   << " cannot be initialized more than once");
    lt.Initialize(options_, params_);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize local transporter implementation.
 */
void IntegrationSingleton::finalize_local_impl()
{
    CELER_LOG(debug) << "Destroying local state";
    auto& lt = this->local_offload();
    CELER_VALIDATE(lt,
                   << "local thread " << G4Threading::G4GetThreadId() + 1
                   << " cannot be finalized more than once");
    params_.timer()->RecordActionTime(lt.GetActionTime());
    lt.Finalize();
}

//---------------------------------------------------------------------------//
/*!
 * Finalize shared params implementation.
 */
void IntegrationSingleton::finalize_shared_impl()
{
    CELER_LOG(status) << "Finalizing Celeritas";
    CELER_VALIDATE(params_, << "params cannot be finalized more than once");
    params_.timer()->RecordTotalTime(get_time_());
    params_.Finalize();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
