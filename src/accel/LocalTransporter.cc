//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalTransporter.cc
//---------------------------------------------------------------------------//
#include "LocalTransporter.hh"

#include <csignal>
#include <type_traits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ParticleDefinition.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/Convert.geant.hh"
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
                                   SharedParams const& params)
    : auto_flush_(options.max_num_tracks)
    , max_steps_(options.max_steps)
    , dump_primaries_{params.offload_writer()}
    , hit_manager_{params.hit_manager()}
{
    CELER_VALIDATE(params,
                   << "Celeritas SharedParams was not initialized before "
                      "constructing LocalTransporter (perhaps the master "
                      "thread did not call BeginOfRunAction?");
    particles_ = params.Params()->particle();

    // Thread ID is -1 when running serially
    auto thread_id = G4Threading::IsMultithreadedApplication()
                         ? G4Threading::G4GetThreadId()
                         : 0;
    CELER_VALIDATE(thread_id >= 0,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is invalid (perhaps LocalTransporter is being built "
                      "on a non-worker thread?)");
    CELER_VALIDATE(
        static_cast<size_type>(thread_id) < params.Params()->max_streams(),
        << "Geant4 ThreadID (" << thread_id
        << ") is out of range for the reported number of worker threads ("
        << params.Params()->max_streams() << ")");

    StepperInput inp;
    inp.params = params.Params();
    inp.stream_id = StreamId{static_cast<size_type>(thread_id)};
    inp.num_track_slots = options.max_num_tracks;
    inp.sync = options.sync;

    if (celeritas::device())
    {
        step_ = std::make_shared<Stepper<MemSpace::device>>(std::move(inp));
    }
    else
    {
        step_ = std::make_shared<Stepper<MemSpace::host>>(std::move(inp));
    }

    // Set stream ID for finalizing
    hit_manager_.finalizer(HMFinalizer{inp.stream_id});
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID at the start of an event.
 */
void LocalTransporter::SetEventId(int id)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id >= 0);
    event_id_ = EventId(id);
    track_counter_ = 0;
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

    track.particle_id = particles_->find(
        PDGNumber{g4track.GetDefinition()->GetPDGEncoding()});
    track.energy = units::MevEnergy{
        convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV)};

    CELER_VALIDATE(track.particle_id,
                   << "cannot offload '"
                   << g4track.GetDefinition()->GetParticleName()
                   << "' particles");

    track.position = convert_from_geant(g4track.GetPosition(), CLHEP::cm);
    track.direction = convert_from_geant(g4track.GetMomentumDirection(), 1);
    track.time = convert_from_geant(g4track.GetGlobalTime(), CLHEP::s);

    // TODO: Celeritas track IDs are independent from Geant4 track IDs, since
    // they must be sequential from zero for a given event. We may need to save
    // (and share with sensitive detectors!) a map of track IDs for calling
    // back to Geant4.
    track.track_id = TrackId{track_counter_++};
    track.event_id = event_id_;

    buffer_.push_back(track);
    if (buffer_.size() >= auto_flush_)
    {
        // TODO: maybe only run one iteration? But then make sure that Flush
        // still transports active tracks to completion.
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

    CELER_LOG_LOCAL(info) << "Transporting " << buffer_.size()
                          << " tracks from event " << event_id_.unchecked_get()
                          << " with Celeritas";

    if (dump_primaries_)
    {
        // Write offload particles if user requested
        (*dump_primaries_)(buffer_);
    }

    // Abort cleanly for interrupt and user-defined signals
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
                   << "some offloaded tracks were not flushed");

    // Reset all data
    CELER_LOG_LOCAL(debug) << "Resetting local transporter";
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Clear thread-local hit manager on destruction.
 */
void LocalTransporter::HMFinalizer::operator()(SPHitManger& hm) const
{
    if (hm)
    {
        if (this->sid)
        {
            hm->finalize(this->sid);
        }
        else
        {
            CELER_LOG_LOCAL(warning) << "Not finalizing hit manager because "
                                        "stream ID is unset";
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
