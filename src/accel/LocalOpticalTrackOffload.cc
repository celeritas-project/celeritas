//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalTrackOffload.cc
//---------------------------------------------------------------------------//
#include "LocalOpticalTrackOffload.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4EventManager.hh>
#include <G4MTRunManager.hh>

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/Transporter.hh"

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Offload Geant4 optical photon tracks to Celeritas
 */
LocalOpticalTrackOffload::LocalOpticalTrackOffload(SetupOptions const& options,
                                                   SharedParams& params)
{
    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local optical track offload when "
                      "Celeritas "
                      "offloading is disabled");

    // Save a pointer to the optical transporter
    transport_ = params.optical_problem_loaded().transporter;

    // Save a pointer to the direct generator action to insert tracks
    direct_gen_
        = std::dynamic_pointer_cast<optical::DirectGeneratorAction const>(
            params.optical_problem_loaded().generator);
    CELER_VALIDATE(direct_gen_, << "invalid optical DirectGeneratorAction");

    CELER_ASSERT(transport_);
    CELER_ASSERT(transport_->params());

    auto const& optical_params = *transport_->params();

    // Check the thread ID and MT model
    validate_geant_threading(optical_params.max_streams());

    CELER_EXPECT(options.optical);
    auto const& capacity = options.optical->capacity;
    auto_flush_ = capacity.primaries;

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
    if (optical_params.aux_reg())
    {
        state_->aux() = std::make_shared<AuxStateVec>(
            *optical_params.aux_reg(), memspace, stream_id, capacity.tracks);
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
    if constexpr (CELERITAS_RESEED == CELERITAS_RESEED_TRACKSLOT)
    {
        if (!(G4Threading::IsMultithreadedApplication()
              && G4MTRunManager::SeedOncePerCommunication()))
        {
            // Since Geant4 schedules events dynamically, reseed the Celeritas
            // RNGs using the Geant4 event ID for reproducibility. This
            // guarantees that an event can be reproduced given the event ID.
            state_->reseed(transport_->params()->rng(),
                           id_cast<UniqueEventId>(id));
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Buffer optical tracks.
 */
void LocalOpticalTrackOffload::Push(G4Track& g4track)
{
    CELER_EXPECT(*this);

    ++num_pushed_;

    CELER_EXPECT(g4track.GetDefinition());
    CELER_EXPECT(g4track.GetDefinition()->GetParticleName() == "opticalphoton");

    // Convert Geant4 track to optical::TrackInitializer
    TrackData init;

    init.energy = units::MevEnergy(
        convert_from_geant(g4track.GetKineticEnergy(), CLHEP::MeV));

    init.position = convert_from_geant(g4track.GetPosition(), CLHEP::cm);

    init.direction = convert_from_geant(g4track.GetMomentumDirection(), 1);

    init.time = convert_from_geant(g4track.GetGlobalTime(), CLHEP::second);
    init.polarization = convert_from_geant(g4track.GetPolarization(), 1);

    ScopedProfiling profile_this{"push"};

    buffer_.push_back(init);

    if (buffer_.size() >= auto_flush_)
    {
        this->Flush();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Flush buffered optical photon tracks.
 */
void LocalOpticalTrackOffload::Flush()
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

    // Inject buffered tracks into optical state for transport
    direct_gen_->insert(*state_, make_span(buffer_));

    // Transport tracks
    (*transport_)(*state_);

    ++num_flushed_;
    buffer_.clear();
}
//---------------------------------------------------------------------------//
auto LocalOpticalTrackOffload::GetActionTime() const -> MapStrDbl
{
    CELER_EXPECT(*this);
    // Return action times from optical transport
    return transport_->get_action_times(*state_->aux());
}

//---------------------------------------------------------------------------//
/*!
 * Finalize the local optical track offload state
 */
void LocalOpticalTrackOffload::Finalize()
{
    CELER_EXPECT(*this);

    CELER_VALIDATE(buffer_.empty(),
                   << buffer_.size() << " optical tracks were not flushed");

    CELER_LOG(info) << "Finalizing Celeritas after " << num_pushed_
                    << " optical tracks pushed over " << num_flushed_
                    << " flushes";

    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
