//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <algorithm>
#include <csignal>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4EventManager.hh>
#include <G4MTRunManager.hh>
#include <G4ParticleDefinition.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "corecel/sys/TraceCounter.hh"
#include "corecel/sys/TracingSession.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantSd.hh"  // IWYU pragma: keep
#include "celeritas/ext/GeantTrackReconstruction.hh"
#include "celeritas/ext/GeantTrackView.hh"
#include "celeritas/ext/detail/HitProcessor.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/global/CoreParams.hh"  // IWYU pragma: keep
#include "celeritas/global/PrimaryCapacity.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/io/OffloadWriter.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/Primary.hh"

#include "SetupOptions.hh"
#include "SharedParams.hh"

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

#define CELER_VALIDATE_OR_KILL_ACTIVE(COND, MSG, STEPPER) \
    do \
    { \
        if (CELER_UNLIKELY(!(COND))) \
        { \
            std::ostringstream celer_runtime_msg_; \
            celer_runtime_msg_ MSG; \
            if (nonfatal_flush()) \
            { \
                CELER_LOG_LOCAL(error) << celer_runtime_msg_.str(); \
                (STEPPER).kill_active(); \
            } \
            else \
            { \
                CELER_RUNTIME_FAIL( \
                    ::celeritas::RuntimeError::validate_err_str, \
                    celer_runtime_msg_.str(), \
                    #COND); \
            } \
        } \
    } while (0)
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with shared (MT) params.
 */
LocalTransporter::LocalTransporter(SetupOptions const& options,
                                   SharedParams& params)
    : dump_primaries_{params.offload_writer()}
    , max_step_iters_(options.max_step_iters)
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
    validate_geant_threading(params.Params()->sizes().streams);

    // Create hit processor on the local thread so that it's deallocated when
    // this object is destroyed
    auto stream_id = id_cast<StreamId>(get_geant_thread_id());
    if (auto const& hit_manager = params.hit_manager())
    {
        hit_processor_ = hit_manager->make_local_processor(stream_id);
        track_reconstruction_ = hit_processor_->track_reconstruction();
    }
    if (!track_reconstruction_)
    {
        using VecConstPD = GeantTrackReconstruction::VecParticle;
        auto const& offload = params.OffloadParticles();
        track_reconstruction_ = std::make_shared<GeantTrackReconstruction>(
            VecConstPD(offload.begin(), offload.end()),
            GeantTrackReconstruction::make_g4step());
    }

    // Create stepper
    StepperInput inp;
    inp.params = params.Params();
    inp.stream_id = stream_id;
    inp.actions = params.actions();

    if (celeritas::device())
    {
        step_ = std::make_shared<Stepper<MemSpace::device>>(std::move(inp));
    }
    else
    {
        step_ = std::make_shared<Stepper<MemSpace::host>>(std::move(inp));
    }

    // Save optical pointers if available, for diagnostics
    optical_ = params.problem_loaded().optical_collector;

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
    CELER_EXPECT(id != event_id_);

    event_id_ = id;
    ++run_accum_.events;

    if constexpr (CELERITAS_RESEED == CELERITAS_RESEED_TRACKSLOT)
    {
        if (!(G4Threading::IsMultithreadedApplication()
              && G4MTRunManager::SeedOncePerCommunication()))
        {
            // Since Geant4 schedules events dynamically, reseed the Celeritas
            // RNGs using the Geant4 event ID for reproducibility. This
            // guarantees that an event can be reproduced given the event ID.
            step_->reseed(id_cast<UniqueEventId>(event_id_));
        }
    }

    // Initialize Geant4 event reconstruction and primary ID mapping
    track_reconstruction_->init_event();
}

//---------------------------------------------------------------------------//
/*!
 * Stage buffered primaries for transport.
 *
 * This overlaps primary staging, including the device H2D copy, with later
 * Geant4 \c Push calls. It does not start GPU transport: same-stream ordering
 * guarantees that when stepping begins later, kernels observe the copied
 * primaries. Only one staged batch is currently supported.
 */
void LocalTransporter::stage_buffered_primaries(StepperResult const& prior)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(step_->num_buffered_primaries() > 0);
    CELER_EXPECT(buffered_accum_.primaries == step_->num_buffered_primaries());
    CELER_EXPECT(step_->staged_primaries().empty());
    CELER_EXPECT(staged_accum_.empty());

    CELER_VALIDATE(step_->num_buffered_primaries()
                       <= this->available_primary_capacity(prior),
                   << "primary buffer of size "
                   << step_->num_buffered_primaries()
                   << " exceeds available initializer capacity "
                   << step_->initializer_capacity() << " with " << prior.queued
                   << " queued and " << prior.alive << " alive tracks");

    step_->stage_primaries();
    staged_accum_ = std::exchange(buffered_accum_, {});
}

//---------------------------------------------------------------------------//
/*!
 * Calculate primary capacity after initialization and secondary reservation.
 *
 * Primaries and queued initializers are processed before secondaries are
 * appended. Primaries that fill vacant track slots therefore need no queue
 * space at the end of the step. Any remaining primaries must leave room for
 * the secondary stack, capped at the initializer capacity so configurations
 * with a larger secondary allocation can still admit primaries that fill
 * vacant track slots.
 */
size_type LocalTransporter::available_primary_capacity(
    StepperResult const& prior) const
{
    CELER_EXPECT(*this);

    detail::PrimaryCapacityInput input;
    input.track_slots = step_->state().size();
    input.initializer_capacity = step_->initializer_capacity();
    input.secondary_capacity = step_->secondary_capacity();
    input.queued = prior.queued;
    input.alive = prior.alive;
    return detail::calc_primary_capacity(input);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a step, submitting staged primaries if present.
 */
void LocalTransporter::launch_step()
{
    CELER_EXPECT(*this);
    CELER_EXPECT(!step_->valid());
    CELER_EXPECT(in_flight_accum_.empty());

    auto const staged_primaries = step_->staged_primaries();
    bool const has_staged = !staged_primaries.empty();
    CELER_EXPECT(has_staged || transport_active_);
    if (has_staged && !transport_active_)
    {
        // A batch submitted into an empty state starts a new transport epoch;
        // adding a batch to active tail tracks continues the existing epoch.
        step_iters_ = 0;
    }
    if (has_staged)
    {
        CELER_ASSERT(staged_accum_.primaries == staged_primaries.size());

        if (run_accum_.flushes == 0)
        {
            CELER_LOG_LOCAL(status)
                << R"(Executing the first Celeritas stepping loop)";
        }
        if (celeritas::device())
        {
            CELER_LOG_LOCAL(debug)
                << "Transporting " << staged_accum_.primaries << " tracks ("
                << units::ClhepEnergy{staged_accum_.energy}
                << " cumulative kinetic energy) from event " << event_id_
                << " with Celeritas";
        }
        if (staged_accum_.lost_primaries > 0)
        {
            CELER_LOG_LOCAL(info)
                << "Lost " << units::ClhepEnergy{staged_accum_.lost_energy}
                << " cumulative kinetic energy from "
                << staged_accum_.lost_primaries
                << " primaries that started outside the geometry in event "
                << event_id_;
        }
        if (dump_primaries_)
        {
            std::vector<Primary> dump_buffer(staged_primaries.begin(),
                                             staged_primaries.end());
            (*dump_primaries_)(dump_buffer);
        }
    }

    step_->async();
    CELER_ENSURE(step_->valid());
    CELER_ENSURE(step_->staged_primaries().empty());
    if (has_staged)
    {
        in_flight_accum_ = std::exchange(staged_accum_, {});
        CELER_ENSURE(staged_accum_.empty());
        CELER_ENSURE(in_flight_accum_.primaries == staged_primaries.size());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Consume the result from the current step.
 */
auto LocalTransporter::complete_step() -> StepperResult
{
    CELER_EXPECT(*this);
    CELER_EXPECT(step_->valid());

    auto result = step_->get();
    ++step_iters_;
    transport_active_ = static_cast<bool>(result);
    run_accum_.steps += result.active;
    trace(result);

    if (transport_active_)
    {
        if (step_->staged_primaries().empty())
        {
            CELER_VALIDATE_OR_KILL_ACTIVE(
                step_iters_ < max_step_iters_,
                << "number of step iterations exceeded the allowed maximum ("
                << max_step_iters_ << ")",
                *step_);
        }
        else
        {
            // kill_active rejects staged input, so abort rather than discard a
            // successor batch while recovering from excessive iterations.
            CELER_VALIDATE(
                step_iters_ < max_step_iters_,
                << "number of step iterations exceeded the allowed maximum ("
                << max_step_iters_ << ")");
        }
    }

    if (!in_flight_accum_.empty())
    {
        ++run_accum_.flushes;
        run_accum_.primaries += in_flight_accum_.primaries;
        run_accum_.lost_primaries += in_flight_accum_.lost_primaries;
        in_flight_accum_ = {};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Advance ready transport without blocking.
 */
void LocalTransporter::advance_if_ready()
{
    CELER_EXPECT(*this);

    if (!step_->valid())
    {
        if (!step_->staged_primaries().empty())
        {
            // Submit queued input immediately whenever the Stepper is idle.
            this->launch_step();
        }
        return;
    }
    if (!step_->ready())
    {
        return;
    }

    // Polling consumes at most one result and refills the stream when needed.
    auto result = this->complete_step();
    // Launch staged input even if the preceding transport just became idle.
    if (result || !step_->staged_primaries().empty())
    {
        this->launch_step();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Advance existing transport by one step.
 */
auto LocalTransporter::advance_transport() -> StepperResult
{
    CELER_EXPECT(*this);
    CELER_EXPECT(!step_->valid());

    this->launch_step();
    return this->complete_step();
}

//---------------------------------------------------------------------------//
/*!
 * Advance transport until the producer buffer fits in the initializer queue.
 *
 * Active tracks may remain when this returns. The completed result guarantees
 * that start-of-step initialization will reduce the existing initializer queue
 * and producer buffer enough to preserve the maximum usable capacity for
 * secondaries generated by the next step.
 */
auto LocalTransporter::wait_for_initializer_capacity() -> StepperResult
{
    CELER_EXPECT(*this);
    CELER_EXPECT(step_->valid());
    CELER_EXPECT(step_->num_buffered_primaries() > 0);
    CELER_EXPECT(step_->staged_primaries().empty());

    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};

    // Consume the current result, then advance only as far as needed to admit
    // the producer while retaining secondary capacity after initialization.
    auto track_counts = this->complete_step();
    size_type const init_capacity = step_->initializer_capacity();
    size_type const secondary_capacity = step_->secondary_capacity();
    auto primaries_fit = [this](StepperResult const& result) {
        return step_->num_buffered_primaries()
               <= this->available_primary_capacity(result);
    };
    while (!primaries_fit(track_counts))
    {
        CELER_VALIDATE(
            track_counts,
            << "primary buffer of size " << step_->num_buffered_primaries()
            << " exceeds available initializer capacity " << init_capacity
            << " with " << track_counts.queued << " queued and "
            << track_counts.alive << " alive tracks while reserving up to "
            << std::min(secondary_capacity, init_capacity)
            << " initializers for secondaries");

        // A false result means the queue is already empty: failure above then
        // avoids looping when the producer cannot fit an empty core state.
        track_counts = this->advance_transport();
        CELER_VALIDATE_OR_KILL_ACTIVE(
            !interrupted(), << "caught interrupt signal", *step_);
    }
    return track_counts;
}

//---------------------------------------------------------------------------//
/*!
 * Complete all transport without consuming the producer buffer.
 */
void LocalTransporter::drain_transport()
{
    CELER_EXPECT(*this);
    CELER_EXPECT(step_->valid());
    CELER_EXPECT(step_->staged_primaries().empty());

    /*!
     * Abort cleanly for interrupt and user-defined (i.e., job manager)
     * signals.
     *
     * \todo The signal handler is \em not thread safe. We may need to set an
     * atomic/volatile bit so all local transporters abort.
     */
    ScopedSignalHandler interrupted{SIGINT, SIGUSR2};

    auto track_counts = this->complete_step();
    while (track_counts)
    {
        track_counts = this->advance_transport();
        CELER_VALIDATE_OR_KILL_ACTIVE(
            !interrupted(), << "caught interrupt signal", *step_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 track and add it to the Stepper producer buffer.
 */
void LocalTransporter::Push(G4Track& g4track)
{
    CELER_EXPECT(*this);

    ScopedProfiling profile_this{"push"};
    this->advance_if_ready();

    // Always check the event ID when pushing the first EM track, since the
    // GeantTrackReconstruction needs to be initialized before we "acquire" the
    // track
    if (CELER_UNLIKELY(step_->num_buffered_primaries() == 0))
    {
        if (CELER_UNLIKELY(!event_manager_))
        {
            // Cache the event manager
            event_manager_ = G4EventManager::GetEventManager();
            CELER_ASSERT(event_manager_);
        }

        G4Event const* event = event_manager_->GetConstCurrentEvent();
        CELER_ASSERT(event);
        auto event_id = event->GetEventID();
        CELER_ASSERT(event_id >= 0);
        if (event_id_ != event_id)
        {
            // Reseed (if applicable) and reset the track reconstruction
            this->InitializeEvent(event_id);
        }
    }
    CELER_ASSERT(event_id_ >= 0);

    GeantTrackView gtv{g4track};
    if (!is_inside(bbox_,
                   static_array_cast<real_type>(native_value_from(gtv.pos()))))
    {
        // Primary may have been created outside the GPU geometry extent (which
        // may not match the generator's extent *or* the CPU geant4 extent)
        CELER_LOG_LOCAL(error)
            << "Discarding track outside world bounds: " << gtv.energy()
            << " from " << gtv.particle().name() << " at " << gtv.pos()
            << " along " << gtv.dir();

        buffered_accum_.lost_energy += gtv.energy().value();
        ++buffered_accum_.lost_primaries;
        return;
    }

    Primary track;

    track.energy = gtv.energy();
    track.particle_id = particles_->find(gtv.particle().pdg());
    track.position = static_array_cast<real_type>(native_value_from(gtv.pos()));
    track.direction = static_array_cast<real_type>(gtv.dir());
    track.time = static_cast<real_type>(native_value_from(gtv.time()));
    track.weight = gtv.weight();
    // Generate Celeritas-specific PrimaryID and capture user info
    track.primary_id = track_reconstruction_->acquire(g4track);

    CELER_VALIDATE(track.particle_id,
                   << "cannot offload '" << gtv.particle().name()
                   << "' particles");

    /*!
     * \todo Eliminate event ID from primary.
     */
    track.event_id = EventId{0};

    step_->push_primary(track);
    ++buffered_accum_.primaries;
    buffered_accum_.energy += gtv.energy().value();
    if (step_->num_buffered_primaries() == step_->primary_capacity())
    {
        if (celeritas::device())
        {
            if (!step_->staged_primaries().empty())
            {
                // Submit a staged successor before using the full producer as
                // the next staged batch.
                if (step_->valid())
                {
                    // complete_step stores continuation in transport_active_,
                    // which launch_step uses after the result is discarded.
                    static_cast<void>(this->complete_step());
                }
                this->launch_step();
            }
            StepperResult prior;
            if (step_->valid())
            {
                // Preserve active tracks while making room for this producer
                // batch and the next step's secondaries.
                prior = this->wait_for_initializer_capacity();
            }
            this->stage_buffered_primaries(prior);
            this->launch_step();
        }
        else
        {
            this->Flush();
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Transport all buffered tracks and produced secondaries.
 */
void LocalTransporter::Flush()
{
    CELER_EXPECT(*this);

    bool const has_buffered = step_->num_buffered_primaries() > 0;
    bool const has_staged = !step_->staged_primaries().empty();
    // Rejected primaries have no Stepper state but still need loss accounting.
    if (!step_->valid() && !has_staged && !has_buffered
        && buffered_accum_.lost_primaries == 0)
    {
        return;
    }

    ScopedProfiling profile_this("flush");

    if (!step_->staged_primaries().empty())
    {
        // Preserve an already staged successor by submitting it first.
        if (step_->valid())
        {
            // complete_step stores continuation in transport_active_, which
            // launch_step uses after the result is discarded.
            static_cast<void>(this->complete_step());
        }
        this->launch_step();
    }
    if (step_->num_buffered_primaries() > 0)
    {
        // A partial producer follows the same admission rule as a full batch.
        StepperResult prior;
        if (step_->valid())
        {
            prior = this->wait_for_initializer_capacity();
        }
        this->stage_buffered_primaries(prior);
        this->launch_step();
    }
    if (step_->valid())
    {
        this->drain_transport();
    }

    if (buffered_accum_.lost_primaries > 0)
    {
        CELER_ASSERT(buffered_accum_.primaries == 0);
        CELER_LOG_LOCAL(info)
            << "Lost " << units::ClhepEnergy{buffered_accum_.lost_energy}
            << " cumulative kinetic energy from "
            << buffered_accum_.lost_primaries
            << " primaries that started outside the geometry in event "
            << event_id_;
        run_accum_.lost_primaries += buffered_accum_.lost_primaries;
        buffered_accum_ = {};
    }

    if (hit_processor_)
    {
        auto num_hits = hit_processor_->exchange_hits();
        if (num_hits > 0)
        {
            CELER_LOG_LOCAL(debug) << "Reconstituted " << num_hits
                                   << " hits for event " << event_id_;
            run_accum_.hits += num_hits;
        }
    }
    track_reconstruction_->clear();
}

//---------------------------------------------------------------------------//
/*!
 * Number of accepted primaries not yet accounted as transported.
 */
size_type LocalTransporter::GetBufferSize() const
{
    if (!step_)
    {
        return 0;
    }
    CELER_ASSERT(buffered_accum_.primaries == step_->num_buffered_primaries());
    CELER_ASSERT(staged_accum_.primaries == step_->staged_primaries().size());
    return buffered_accum_.primaries + staged_accum_.primaries
           + in_flight_accum_.primaries;
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
    auto const buffer_size = this->GetBufferSize();
    // Submitted-batch accounting may already be clear while active tail tracks
    // or an unconsumed Stepper result still require transport.
    CELER_VALIDATE(
        !step_->valid() && !transport_active_ && buffer_size == 0,
        << "offloaded tracks were not flushed (" << buffer_size
        << " primaries buffered" << (step_->valid() ? ", step in flight" : "")
        << (transport_active_ ? ", transport incomplete" : "") << ")");

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
        msg << " from " << run_accum_.flushes << " flushes with "
            << run_accum_.primaries << " offloaded tracks over "
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
auto LocalTransporter::GetActionTime() const -> MapStrDbl
{
    CELER_EXPECT(*this);

    auto const& action_seq = step_->actions();
    MapStrDbl result = action_seq.get_action_times(step_->state().aux());
    if (optical_)
    {
        // Save optical loop action times
        auto optical_times = optical_->get_action_times(step_->state().aux());
        for (auto&& [label, time] : optical_times)
        {
            // Prefix label to distinguish from core actions
            result["optical::" + label] = time;
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
