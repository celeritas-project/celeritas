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
#include "corecel/sys/ThreadId.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/ext/GeantSd.hh"
#include "celeritas/g4/StateDependent.hh"
#include "accel/LocalOpticalTrackOffload.hh"

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

    auto const& supported = SharedParams::supported_offload_particles();
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
}  // namespace

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
        else if (options_.optical
                 && std::holds_alternative<inp::OpticalDirectGenerator>(
                     options_.optical->generator))

        {
            // offloading direct optical tracks
            CELER_LOG(info) << "Optical track offloading enabled";
            offload = std::make_unique<LocalOpticalTrackOffload>();
        }
        else
        {
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
 * Initialize shared params if needed and this thread's local transporter.
 *
 * This handles both global (master thread) and local (worker thread)
 * initialization. Shared params persist across repeated BeamOn calls until
 * terminal teardown, while local state is recreated for each run as needed.
 *
 * \return Whether local offload state was initialized and should be verified
 */
bool IntegrationSingleton::initialize_offload()
{
    if (G4Threading::IsMasterThread())
    {
        if (!params_)
        {
            failed_setup_ = false;

            ExceptionConverter call_g4exception{"celer.init.global"};
            CELER_TRY_HANDLE(this->initialize_master_impl(), call_g4exception);
            failed_setup_ = call_g4exception.forwarded();

            // Start the run timer
            get_time_ = {};
        }
        else
        {
            CELER_LOG_LOCAL(debug) << "Shared Celeritas state already "
                                      "initialized";
        }
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

    bool initialized = false;
    CELER_TRY_HANDLE(initialized = this->initialize_local_impl(),
                     ExceptionConverter("celer.init.local"));
    return initialized;
}

//---------------------------------------------------------------------------//
/*!
 * Finalize any active local transporter and, on the master thread, shared
 * params.
 */
void IntegrationSingleton::finalize_offload()
{
    if (CELER_UNLIKELY(failed_setup_ || !params_))
    {
        return;
    }

    this->finalize_local_offload();
    this->finalize_shared_offload();
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
 * Set callback to verify thread-local setup at begin_run.
 */
void IntegrationSingleton::set_verify_callback(VerifyCallback cb)
{
    verify_callback_ = std::move(cb);
}

//---------------------------------------------------------------------------//
/*!
 * Register master-thread Geant4 state hook.
 *
 * Worker hooks are thread-local and registered from
 * TrackingManagerConstructor on each worker thread.
 */
void IntegrationSingleton::register_auto_hooks()
{
    if (auto_hooks_active_)
    {
        return;
    }

    // Register master-thread state monitor. This object is owned like the
    // SetupOptionsMessenger: Geant4 state notifications are callback-only and
    // do not control ownership.
    master_state_dependent_ = std::make_unique<StateDependent>(
        [this](StreamId sid, GeantStateChange change) {
            this->on_state_change(sid, change);
        },
        StateDependent::Mode::lifecycle,
        StateDependent::LifecycleRole::global);
    auto_hooks_active_ = true;
}

//---------------------------------------------------------------------------//
/*!
 * Drive offload init/finalize from Geant4 state transitions.
 *
 * \c StateDependent filters Geant4 run-manager ordering details before
 * invoking this function.
 */
void IntegrationSingleton::on_state_change(StreamId stream_id,
                                           GeantStateChange change)
{
    switch (change)
    {
        case GeantStateChange::begin_run: {
            bool enable_offload = false;
            CELER_TRY_HANDLE(
                { enable_offload = this->initialize_offload(); },
                ExceptionConverter{"celer.init.auto"});
            if (enable_offload && verify_callback_)
            {
                CELER_TRY_HANDLE(verify_callback_(stream_id),
                                 ExceptionConverter{"celer.init.verify"});
            }
            break;
        }
        case GeantStateChange::end_run:
            CELER_TRY_HANDLE(this->finalize_local_offload(),
                             ExceptionConverter{"celer.finalize.local.auto"});
            break;
        case GeantStateChange::end_program:
            CELER_TRY_HANDLE(this->finalize_offload(),
                             ExceptionConverter{"celer.finalize.auto"});
            break;
        default:
            break;
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
bool IntegrationSingleton::initialize_local_impl()
{
    CELER_EXPECT(!G4Threading::IsMultithreadedApplication()
                 || G4Threading::IsWorkerThread());

    auto& lt = this->local_offload();
    if (lt)
    {
        CELER_LOG_LOCAL(debug)
            << "Local Celeritas state already initialized on thread "
            << G4Threading::G4GetThreadId() + 1;
        return false;
    }
    CELER_LOG(debug) << "Constructing local state";
    lt.Initialize(options_, params_);
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Finalize local transporter implementation.
 */
bool IntegrationSingleton::finalize_local_impl()
{
    auto& lt = this->local_offload();
    if (!lt)
    {
        CELER_LOG_LOCAL(debug)
            << "Local Celeritas state already finalized on thread "
            << G4Threading::G4GetThreadId() + 1;
        return false;
    }
    CELER_LOG(debug) << "Destroying local state";
    params_.timer()->RecordActionTime(lt.GetActionTime());
    lt.Finalize();
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Finalize thread-local offload state owned by this thread.
 */
void IntegrationSingleton::finalize_local_offload()
{
    if (CELER_UNLIKELY(failed_setup_ || !params_))
    {
        return;
    }

    if (params_.mode() == OffloadMode::enabled
        && (!G4Threading::IsMultithreadedApplication()
            || !G4Threading::IsMasterThread()))
    {
        CELER_TRY_HANDLE(
            { static_cast<void>(this->finalize_local_impl()); },
            ExceptionConverter("celer.finalize.local"));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Finalize shared offload state owned by the master thread.
 */
void IntegrationSingleton::finalize_shared_offload()
{
    if (CELER_UNLIKELY(failed_setup_ || !params_))
    {
        return;
    }

    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(this->finalize_shared_impl(),
                         ExceptionConverter("celer.finalize.global"));
    }
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
