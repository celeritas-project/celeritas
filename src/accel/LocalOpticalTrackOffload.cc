//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalTrackOffload.cc
//---------------------------------------------------------------------------//
#include "LocalOpticalTrackOffload.hh"

#include <G4EventManager.hh>
#include <G4MTRunManager.hh>

#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/Transporter.hh"

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 *
 */
LocalOpticalTrackOffload::LocalOpticalTrackOffload(SetupOptions const& options,
                                                   SharedParams& params)
{
    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local optical track offload when "
                      "Celeritas "
                      "offloading is disabled");

    // Check the thread ID and MT model
    validate_geant_threading(params.Params()->max_streams());

    // Save a pointer to the optical transporter
    transport_ = params.optical_transporter();
    CELER_ASSERT(transport_);
    CELER_ASSERT(transport_->params());
    auto const& optical_params = *transport_->params();

    // Number of optical tracks to buffer before offloading
    auto const& capacity = options.optical->capacity;
    auto_flush_ = capacity.tracks;

    auto stream_id = id_cast<StreamId>(get_geant_thread_id());

    // Allocate thread-local state data
    auto memspace = celeritas::device() ? MemSpace::device : MemSpace::host;
    if (memspace == MemSpace::device)
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::device>>(
            optical_params, stream_id, capacity.tracks);
    }
    else
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::host>>(
            optical_params, stream_id, capacity.tracks);
    }

    // Allocate auxiliary data
    if (params.Params()->aux_reg())
    {
        state_->aux() = std::make_shared<AuxStateVec>(
            *params.Params()->aux_reg(), memspace, stream_id, capacity.tracks);
    }

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize with options and shared data.
 */
void LocalOpticalTrackOffload::Initialize(SetupOptions const& options,
                                          SharedParams& params)
{
    *this = LocalOpticalTrackOffload(options, params);
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalOpticalTrackOffload::InitializeEvent(int id)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id >= 0);

    event_id_ = id_cast<UniqueEventId>(id);
    if (!(G4Threading::IsMultithreadedApplication()
          && G4MTRunManager::SeedOncePerCommunication()))
    {
        // Since Geant4 schedules events dynamically, reseed the Celeritas
        //  RNGs
        // using the Geant4 event ID for reproducibility. This guarantees
        // that
        // an event can be reproduced given the event ID.
        state_->reseed(transport_->params()->rng(), id_cast<UniqueEventId>(id));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Buffer distribution data for generating optical photons.
 */
void LocalOpticalTrackOffload::Push(G4Track& g4track)
{
    CELER_EXPECT(*this);
    ++num_pushed_;
    TrackData init;

    CELER_EXPECT(g4track.GetDefinition());
    CELER_EXPECT(g4track.GetDefinition()->GetParticleName() == "opticalphoton");

    // Energy: convert Geant4 energy [MeV] to Celeritas MevEnergy
    init.energy = units::MevEnergy{g4track.GetTotalEnergy() / CLHEP::MeV};

    // Position: Geant4 uses mm; Celeritas uses cm
    auto const& pos = g4track.GetPosition();
    init.position
        = Real3{pos.x() / CLHEP::cm, pos.y() / CLHEP::cm, pos.z() / CLHEP::cm};

    auto const& dir = g4track.GetMomentumDirection();
    init.direction = Real3{dir.x(), dir.y(), dir.z()};

    // Polarization: directly from G4
    auto const& pol = g4track.GetPolarization();
    init.polarization = Real3{pol.x(), pol.y(), pol.z()};

    // Time: Geant4 uses ns; Celeritas uses seconds
    init.time = g4track.GetGlobalTime() / CLHEP::s;

    ScopedProfiling profile_this{"push"};

    buffer_.push_back(init);
    pending_tracks_++;

    if (pending_tracks_ >= auto_flush_)
    {
        this->Flush();
    }
}
//---------------------------------------------------------------------------//
/*!
 * Generate and transport optical photons from the buffered distribution data.
 */
void LocalOpticalTrackOffload::Flush()
{
    CELER_EXPECT(*this);
    if (buffer_.empty())
    {
        return;
    }

    ScopedProfiling profile_this("flush");

    //! \todo Duplicated in \c LocalTransporter
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

    // Inject buffered tracks into optical state
    ++num_flushed_;
    //  state_->insert_primaries(make_span(buffer_));
    // ToDo Skipping optical transport (WIP)
    buffer_.clear();
    pending_tracks_ = 0;

    // Generate optical photons and transport to completion
    // (*transport_)(*state_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the accumulated action times.
 */
auto LocalOpticalTrackOffload::GetActionTime() const -> MapStrDbl
{
    CELER_EXPECT(*this);
    return transport_->get_action_times(*state_->aux());
}
//---------------------------------------------------------------------------//
/*!
 * Clear local data.
 */
void LocalOpticalTrackOffload::Finalize()
{
    CELER_EXPECT(*this);

    CELER_VALIDATE(buffer_.empty(),
                   << pending_tracks_ << " optical tracks were not flushed");

    CELER_LOG(info) << "Finalizing Celeritas after " << num_pushed_
                    << " optical tracks pushed (over " << num_flushed_
                    << " ) flushes";

    // Reset all data
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
