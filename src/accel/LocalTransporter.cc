//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <csignal>
#include <string>
#include <type_traits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MTRunManager.hh>
#include <G4ParticleDefinition.hh>
#include <G4Threading.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/Config.hh"

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/g4/Convert.geant.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/global/ActionSequence.hh"
#include "celeritas/io/EventWriter.hh"
#include "celeritas/io/RootEventWriter.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"  // IWYU pragma: keep

#include "SetupOptions.hh"
#include "SharedParams.hh"

#include "detail/HitManager.hh"
#include "detail/OffloadWriter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared (MT) params.
 */
LocalTransporter::LocalTransporter(SetupOptions const& options,
                                   SharedParams& params)
    : auto_flush_(options.auto_flush ? options.auto_flush
                                     : options.max_num_tracks)
    , max_steps_(options.max_steps)
    , dump_primaries_{params.offload_writer()}
{
    CELER_VALIDATE(params,
                   << "Celeritas SharedParams was not initialized before "
                      "constructing LocalTransporter (perhaps the master "
                      "thread did not call BeginOfRunAction?");
    particles_ = params.Params()->particle();

    auto thread_id = get_geant_thread_id();
    CELER_VALIDATE(thread_id >= 0,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is invalid (perhaps LocalTransporter is being built "
                      "on a non-worker thread?)");
    CELER_VALIDATE(
        static_cast<size_type>(thread_id) < params.Params()->max_streams(),
        << "Geant4 ThreadID (" << thread_id
        << ") is out of range for the reported number of worker threads ("
        << params.Params()->max_streams() << ")");

    // Check that OpenMP and Geant4 threading models don't collide
    if (CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK && !celeritas::device()
        && G4Threading::IsMultithreadedApplication())
    {
        auto msg = CELER_LOG_LOCAL(warning);
        msg << "Using multithreaded Geant4 with Celeritas track-level OpenMP "
               "parallelism";
        if (std::string const& nt_str = celeritas::getenv("OMP_NUM_THREADS");
            !nt_str.empty())
        {
            msg << "(OMP_NUM_THREADS=" << nt_str
                << "): CPU threads may be oversubscribed";
        }
        else
        {
            msg << ": forcing 1 Celeritas thread to Geant4 thread";
#ifdef _OPENMP
            omp_set_num_threads(1);
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }
    }

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        /*!
         * \todo Add support for Geant4 navigation wrapper, which requires
         * calling \c state.ref().geometry.reset() on the local transporter
         * thread due to thread-allocated navigator data.
         */
        CELER_NOT_IMPLEMENTED(
            "offloading when using Celeritas Geant4 navigation wrapper");
    }

    // Create hit processor on the local thread so that it's deallocated when
    // this object is destroyed
    StreamId stream_id{static_cast<size_type>(thread_id)};
    if (auto const& hit_manager = params.hit_manager())
    {
        hit_processor_ = hit_manager->make_local_processor(stream_id);
    }

    // Create stepper
    StepperInput inp;
    inp.params = params.Params();
    inp.stream_id = stream_id;
    inp.num_track_slots = options.max_num_tracks;
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
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalTransporter::InitializeEvent(int id)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id >= 0);

    event_id_ = EventId(id);
    track_counter_ = 0;

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
void LocalTransporter::Push(G4Track const& g4track)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(event_id_);

    Primary track;

    PDGNumber const pdg{g4track.GetDefinition()->GetPDGEncoding()};
    track.particle_id = particles_->find(pdg);
    track.energy = units::MevEnergy(
        convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV));

    CELER_VALIDATE(track.particle_id,
                   << "cannot offload '"
                   << g4track.GetDefinition()->GetParticleName()
                   << "' particles");

    track.position = convert_from_geant(g4track.GetPosition(), clhep_length);
    track.direction = convert_from_geant(g4track.GetMomentumDirection(), 1);
    track.time = convert_from_geant(g4track.GetGlobalTime(), clhep_time);

    if (CELER_UNLIKELY(g4track.GetWeight() != 1.0))
    {
        //! \todo Non-unit weights: see issue #1268
        CELER_LOG(error) << "incoming track (PDG " << pdg.get()
                         << ", track ID " << g4track.GetTrackID()
                         << ") has non-unit weight " << g4track.GetWeight();
    }

    /*!
     * \todo Celeritas track IDs are independent from Geant4 track IDs, since
     * they must be sequential from zero for a given event. We may need to save
     * (and share with sensitive detectors!) a map of track IDs for calling
     * back to Geant4.
     */
    track.track_id = TrackId{track_counter_++};
    track.event_id = event_id_;

    buffer_.push_back(track);
    if (buffer_.size() >= auto_flush_)
    {
        /*!
         * \todo Maybe only run one iteration? But then make sure that Flush
         * still transports active tracks to completion.
         */
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
    if (celeritas::device())
    {
        CELER_LOG_LOCAL(info)
            << "Transporting " << buffer_.size() << " tracks from event "
            << event_id_.unchecked_get() << " with Celeritas";
    }

    if (dump_primaries_)
    {
        // Write offload particles if user requested
        (*dump_primaries_)(buffer_);
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
    buffer_.clear();

    size_type step_iters = 1;

    while (track_counts)
    {
        CELER_VALIDATE(step_iters < max_steps_,
                       << "number of step iterations exceeded the allowed "
                          "maximum ("
                       << max_steps_ << ")");

        track_counts = (*step_)();
        ++step_iters;

        CELER_VALIDATE(!interrupted(), << "caught interrupt signal");
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

    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting local transporter";
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
        auto const& action_ptrs = action_seq.actions();
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
}  // namespace celeritas
