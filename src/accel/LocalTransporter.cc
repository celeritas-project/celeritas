//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <csignal>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4EventManager.hh>
#include <G4MTRunManager.hh>
#include <G4ParticleDefinition.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/TraceCounter.hh"
#include "corecel/sys/TracingSession.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantSd.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/ext/detail/HitProcessor.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/io/EventWriter.hh"
#include "celeritas/io/JsonEventWriter.hh"
#include "celeritas/io/RootEventWriter.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

#include "SetupOptions.hh"
#include "SharedParams.hh"

#include "detail/OffloadWriter.hh"

namespace celeritas
{
namespace
{
bool nonfatal_flush()
{
    static bool const result = [] {
        auto result = getenv_flag("CELER_NONFATAL_FLUSH", false);
        return result.value;
    }();
    return result;
}

bool not_release_build()
{
    std::string_view build_props{cmake::build_type};
    // Instead of searching for `release`, which may not be present in some
    // build systems, see if we have debug or relwithdebinfo.
    if (build_props.find("debug") != std::string_view::npos)
    {
        return true;
    }
    if (build_props.find("relwithdebinfo") != std::string_view::npos)
    {
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------//
//! Trace the number of active, alive, dead, and queued tracks
class TrackCounters
{
  public:
    TrackCounters()
    {
        if (ScopedProfiling::enabled())
        {
            std::string stream_id = std::to_string(get_geant_thread_id());
            active_counter_ = std::string("active-" + stream_id);
            alive_counter_ = std::string("alive-" + stream_id);
            dead_counter_ = std::string("dead-" + stream_id);
            queued_counter_ = std::string("queued-" + stream_id);
        }
    };

    void operator()(StepperResult const& track_counts) const
    {
        trace_counter(active_counter_.c_str(), track_counts.active);
        trace_counter(alive_counter_.c_str(), track_counts.alive);
        trace_counter(dead_counter_.c_str(),
                      track_counts.active - track_counts.alive);
        trace_counter(queued_counter_.c_str(), track_counts.queued);
    }

  private:
    std::string active_counter_;
    std::string alive_counter_;
    std::string dead_counter_;
    std::string queued_counter_;
};

void trace(StepperResult const& track_counts)
{
    static thread_local TrackCounters const trace_;
    trace_(track_counts);
}

#define CELER_VALIDATE_OR_KILL_ACTIVE(COND, MSG, STEPPER)           \
    do                                                              \
    {                                                               \
        if (CELER_UNLIKELY(!(COND)))                                \
        {                                                           \
            std::ostringstream celer_runtime_msg_;                  \
            celer_runtime_msg_ MSG;                                 \
            if (nonfatal_flush())                                   \
            {                                                       \
                CELER_LOG_LOCAL(error) << celer_runtime_msg_.str(); \
                (STEPPER).kill_active();                            \
            }                                                       \
            else                                                    \
            {                                                       \
                CELER_RUNTIME_THROW(                                \
                    ::celeritas::RuntimeError::validate_err_str,    \
                    celer_runtime_msg_.str(),                       \
                    #COND);                                         \
            }                                                       \
        }                                                           \
    } while (0)
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with shared (MT) params.
 */
LocalTransporter::LocalTransporter(SetupOptions const& options,
                                   SharedParams& params)
    : auto_flush_(
          get_default(options, params.Params()->max_streams()).primaries)
    , max_step_iters_(options.max_step_iters)
    , dump_primaries_{params.offload_writer()}
{
    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local transporter when Celeritas "
                      "offloading is disabled");
    CELER_VALIDATE(!options.optical
                       || std::holds_alternative<inp::OpticalEmGenerator>(
                           options.optical->generator),
                   << "invalid optical photon generation mechanism for local "
                      "transporter");

    particles_ = params.Params()->particle();
    CELER_ASSERT(particles_);
    bbox_ = params.bbox();

    // Check the thread ID and MT model
    validate_geant_threading(params.Params()->max_streams());

    // Create hit processor on the local thread so that it's deallocated when
    // this object is destroyed
    auto stream_id = id_cast<StreamId>(get_geant_thread_id());
    if (auto const& hit_manager = params.hit_manager())
    {
        hit_processor_ = hit_manager->make_local_processor(stream_id);
    }

    // Create stepper
    StepperInput inp;
    inp.params = params.Params();
    inp.stream_id = stream_id;
    inp.action_times = options.action_times;

    if (celeritas::device())
    {
        step_ = std::make_shared<Stepper<MemSpace::device>>(std::move(inp));
    }
    else
    {
        step_ = std::make_shared<Stepper<MemSpace::host>>(std::move(inp));
    }

    // Save state for reductions at the end
    params.set_state(stream_id.get(), step_->sp_state());

    // Save optical pointers if available, for diagnostics
    optical_ = params.optical_collector();

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalTransporter::InitializeEvent(int id)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id >= 0);

    event_id_ = id_cast<UniqueEventId>(id);
    ++run_accum_.events;

    if (!(G4Threading::IsMultithreadedApplication()
          && G4MTRunManager::SeedOncePerCommunication()))
    {
        // Since Geant4 schedules events dynamically, reseed the Celeritas RNGs
        // using the Geant4 event ID for reproducibility. This guarantees that
        // an event can be reproduced given the event ID.
        step_->reseed(event_id_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 track to a Celeritas primary and add to buffer.
 */
void LocalTransporter::Push(G4Track& g4track)
{
    CELER_EXPECT(*this);

    ScopedProfiling profile_this{"push"};

    if (Real3 pos = convert_from_geant(g4track.GetPosition(), 1);
        !is_inside(bbox_, pos))
    {
        // Primary may have been created by a particle generator outside the
        // geometry
        double energy
            = convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV);
        CELER_LOG_LOCAL(error)
            << "Discarding track outside world bounds: " << energy
            << " MeV from " << g4track.GetDefinition()->GetParticleName()
            << " at " << pos << " along "
            << convert_from_geant(g4track.GetMomentumDirection(), 1);

        buffer_accum_.lost_energy += energy;
        ++buffer_accum_.lost_primaries;
        return;
    }

    Primary track;

    PDGNumber const pdg{g4track.GetDefinition()->GetPDGEncoding()};
    track.particle_id = particles_->find(pdg);

    // Generate Celeritas-specific PrimaryID
    if (hit_processor_)
    {
        track.primary_id
            = hit_processor_->track_processor().register_primary(g4track);
    }

    track.energy = units::MevEnergy(
        convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV));

    CELER_VALIDATE(track.particle_id,
                   << "cannot offload '"
                   << g4track.GetDefinition()->GetParticleName()
                   << "' particles");

    track.position = convert_from_geant(g4track.GetPosition(), clhep_length);
    track.direction = convert_from_geant(g4track.GetMomentumDirection(), 1);
    track.time = convert_from_geant(g4track.GetGlobalTime(), clhep_time);
    track.weight = g4track.GetWeight();

    /*!
     * \todo Eliminate event ID from primary.
     */
    track.event_id = EventId{0};

    buffer_.push_back(track);
    buffer_accum_.energy += track.energy.value();
    if (buffer_.size() >= auto_flush_)
    {
        this->Flush();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Transport the buffered tracks and all secondaries produced.
 */
void LocalTransporter::Flush()
{
    CELER_EXPECT(*this);
    if (buffer_.empty())
    {
        return;
    }

    ScopedProfiling profile_this("flush");

    if (event_manager_ || !event_id_)
    {
        if (CELER_UNLIKELY(!event_manager_))
        {
            // Save the event manager pointer, thereby marking that
            // *subsequent* events need to have their IDs checked as well
            event_manager_ = G4EventManager::GetEventManager();
            CELER_ASSERT(event_manager_);
        }

        G4Event const* event = event_manager_->GetConstCurrentEvent();
        CELER_ASSERT(event);
        if (event_id_ != id_cast<UniqueEventId>(event->GetEventID()))
        {
            // The event ID has changed: reseed it
            this->InitializeEvent(event->GetEventID());
        }
    }
    CELER_ASSERT(event_id_);

    if (celeritas::device())
    {
        CELER_LOG_LOCAL(debug)
            << "Transporting " << buffer_.size() << " tracks ("
            << buffer_accum_.energy
            << " MeV cumulative kinetic energy) from event "
            << event_id_.unchecked_get() << " with Celeritas";
    }
    if (buffer_accum_.lost_primaries > 0)
    {
        CELER_LOG_LOCAL(info)
            << "Lost " << buffer_accum_.lost_energy
            << " MeV cumulative kinetic energy from "
            << buffer_accum_.lost_primaries
            << " primaries that started outside the geometry in event "
            << event_id_.unchecked_get();
    }

    if (dump_primaries_)
    {
        // Write offload particles if user requested
        (*dump_primaries_)(buffer_);
    }

    if (run_accum_.steps == 0)
    {
        CELER_LOG_LOCAL(status)
            << R"(Executing the first Celeritas stepping loop)";
    }

    /*!
     * Abort cleanly for interrupt and user-defined (i.e., job manager)
     * signals.
     *
     * \todo The signal handler is \em not thread safe. We may need to set an
     * atomic/volatile bit so all local transporters abort.
     */
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};

    // Copy buffered tracks to device and transport the first step
    auto track_counts = (*step_)(make_span(buffer_));
    run_accum_.steps += track_counts.active;
    run_accum_.primaries += buffer_.size();
    run_accum_.lost_primaries += buffer_accum_.lost_primaries;
    trace(track_counts);

    buffer_.clear();
    buffer_accum_ = {};

    size_type step_iters = 1;

    while (track_counts)
    {
        CELER_VALIDATE_OR_KILL_ACTIVE(step_iters < max_step_iters_,
                                      << "number of step iterations exceeded "
                                         "the allowed maximum ("
                                      << max_step_iters_ << ")",
                                      *step_);

        track_counts = (*step_)();
        run_accum_.steps += track_counts.active;
        ++step_iters;
        trace(track_counts);
        CELER_VALIDATE_OR_KILL_ACTIVE(
            !interrupted(), << "caught interrupt signal", *step_);
    }

    if (hit_processor_)
    {
        auto num_hits = hit_processor_->exchange_hits();
        if (num_hits > 0)
        {
            CELER_LOG_LOCAL(debug) << "Reconstituted " << num_hits
                                   << " hits for event " << event_id_.get();
            run_accum_.hits += num_hits;
        }
        hit_processor_->track_processor().end_event();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data.
 *
 * This may need to be executed on the same thread it was created in order to
 * safely deallocate some Geant4 objects under the hood...
 */
void LocalTransporter::Finalize()
{
    CELER_EXPECT(*this);
    CELER_VALIDATE(buffer_.empty(),
                   << "offloaded tracks (" << buffer_.size()
                   << " in buffer) were not flushed");

    std::size_t num_optical_steps{0};
    {
        auto msg = CELER_LOG_LOCAL(info);
        msg << "Finalizing Celeritas after " << run_accum_.steps << " steps";
        if (optical_)
        {
            auto const& state = optical_->optical_state(this->GetState());
            auto const& accum = state.accum();
            num_optical_steps = state.accum().steps;
            msg << " and " << num_optical_steps << " optical steps (over "
                << accum.step_iters << " step iterations)";
        }
        msg << " from " << run_accum_.primaries << " offloaded tracks over "
            << run_accum_.events << " events, generating " << run_accum_.hits
            << " hits";
    }
    if (run_accum_.lost_primaries > 0)
    {
        CELER_LOG_LOCAL(warning)
            << "Lost a total of " << run_accum_.lost_primaries
            << " primaries that started outside the world";
    }
    static bool have_warned_slow{false};
    if (!have_warned_slow && (run_accum_.steps + num_optical_steps > 1000000)
        && (CELERITAS_DEBUG || not_release_build()))
    {
        static std::mutex mu;
        std::lock_guard scoped_lock{mu};
        if (!have_warned_slow)
        {
            CELER_LOG(warning) << "Performance is degraded due to "
                                  "non-optimized build options: "
                               << BuildOutput{};
            have_warned_slow = true;
        }
    }

    if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        // Geant4 navigation states *MUST* be deallocated on the thread in
        // which they're allocated
        auto state = std::dynamic_pointer_cast<CoreState<MemSpace::host>>(
            step_->sp_state());
        CELER_ASSERT(state);
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
        state->ref().geometry.reset();
#endif
    }

    // Flush any remaining performance counters on the worker thread
    TracingSession::flush();

    // Reset all data
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 */
auto LocalTransporter::GetActionTime() const -> MapStrReal
{
    CELER_EXPECT(*this);

    MapStrReal result;
    auto const& action_seq = step_->actions();
    if (action_seq.action_times())
    {
        // Save kernel timing if synchronization is enabled
        auto const& action_ptrs = action_seq.actions().step();
        auto const& time = action_seq.accum_time();

        CELER_ASSERT(action_ptrs.size() == time.size());
        for (auto i : range(action_ptrs.size()))
        {
            result[std::string{action_ptrs[i]->label()}] = time[i];
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Access core state data for user diagnostics.
 */
CoreStateInterface const& LocalTransporter::GetState() const
{
    CELER_EXPECT(*this);

    return step_->state();
}

//---------------------------------------------------------------------------//
/*!
 * Access core state data for user diagnostics.
 */
CoreStateInterface& LocalTransporter::GetState()
{
    CELER_EXPECT(*this);

    // NOTE: the Stepper will be removed in the near term in a major refactor
    // of the shared params and state, so we allow this as a convenience hack
    return const_cast<CoreStateInterface&>(step_->state());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
