//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.cc
//---------------------------------------------------------------------------//
#include "GeantTrackReconstruction.hh"

#include <limits>
#include <mutex>
#include <G4DynamicParticle.hh>
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4ParticleDefinition.hh>
#include <G4Step.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4VProcess.hh>
#include <G4VUserTrackInformation.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
[[maybe_unused]] int get_g4_current_event_id()
{
    auto* evtman = G4EventManager::GetEventManager();
    CELER_ASSERT(evtman);
    auto* evt = evtman->GetConstCurrentEvent();
    if (!evt)
    {
        // Use a different "invalid" event ID from the default g4_event_id_
        return -2;
    }
    return evt->GetEventID();
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Event ID function pointer for unit testing when CELERITAS_DEBUG.
 *
 * When constructing a class instance, if the function pointer is null, it will
 * be set to a function that gets the Geant4 event manager's active event.
 */
GeantTrackReconstruction::EventIdGetter
    GeantTrackReconstruction::get_current_event_id{nullptr};

//---------------------------------------------------------------------------//
/*!
 * Allocate and initialize a valid Geant4 step object.
 *
 * The allocation is done like \c G4SteppingManager constructor but we reset
 * values to invalid ones.
 */
auto GeantTrackReconstruction::make_g4step() -> SPStep
{
    auto step = std::make_shared<G4Step>();

    // Allocate secondary vector, needed to keep some SDs from crashing
    step->NewSecondaryVector();

    // Set invalid values for unsupported SD attributes
    step->SetNonIonizingEnergyDeposit(-std::numeric_limits<double>::infinity());
    for (G4StepPoint* p : {step->GetPreStepPoint(), step->GetPostStepPoint()})
    {
        p->SetStepStatus(fUserDefinedLimit);
        // Time since track was created
        p->SetLocalTime(std::numeric_limits<double>::infinity());
        // Time in rest frame since track was created
        p->SetProperTime(std::numeric_limits<double>::infinity());
        // Speed (TODO: use ParticleView)
        p->SetVelocity(std::numeric_limits<double>::infinity());
        // Safety distance
        p->SetSafety(std::numeric_limits<double>::infinity());
        // Polarization (default to zero)
        p->SetPolarization(G4ThreeVector());
    }

    return step;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with particle definitions for track reconstruction.
 */
GeantTrackReconstruction::GeantTrackReconstruction(
    VecParticle const& particles, SPStep step)
    : step_(std::move(step))
{
    CELER_EXPECT(!particles.empty());
    CELER_EXPECT(step_);

    // Create track for each particle type
    for (G4ParticleDefinition const* pd : particles)
    {
        CELER_ASSERT(pd);
        auto track = std::make_unique<G4Track>(
            new G4DynamicParticle(pd, G4ThreeVector()), 0.0, G4ThreeVector());
        track->SetTrackID(0);
        track->SetParentID(0);
        tracks_.emplace_back(std::move(track));
    }

    // Set the step for all tracks
    for (auto const& track : tracks_)
    {
        track->SetStep(step_.get());
    }

    // Reset event interface used for test mocking
    if constexpr (CELERITAS_DEBUG)
    {
        static std::mutex mu;
        std::scoped_lock lock{mu};

        if (get_current_event_id == nullptr)
        {
            get_current_event_id = get_g4_current_event_id;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Unset the user information for all tracks
 */
GeantTrackReconstruction::~GeantTrackReconstruction()
{
    try
    {
        CELER_LOG(debug) << "Deallocating track reconstruction";
        if (!g4_track_data_.empty())
        {
            CELER_LOG_LOCAL(warning)
                << R"(Geant4 track data was not cleared during the event)";
        }
        this->clear();
    }
    catch (...)  // NOLINT(bugprone-empty-catch)
    {
        // Ignore anything bad that happens while destroying
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear G4Track reconstruction data.
 *
 * This should be done when all Celeritas tracks have been completed, since
 * afterward it will be impossible to reconstruct them.
 *
 * The primary ID offset is saved to ensure consistency when flushing before
 * an event is complete. User information still lent to handed-back tracks is
 * transferred to them; all other user information is deleted.
 */
void GeantTrackReconstruction::clear()
{
    // Set primary id offset
    start_ = start_ + g4_track_data_.size();

    for (auto& track : tracks_)
    {
        // Clear the user information to prevent double deletion:
        // GeantTrackReconstruction owns the track user info
        track->SetUserInformation(nullptr);
    }
    g4_track_data_.clear();

    // Transfer ownership of lent user information to the tracks using it
    for (auto&& [track, info] : lent_)
    {
        auto iter = user_info_.find(info);
        CELER_ASSERT(iter != user_info_.end());
        // The track now owns (and will delete) the user information
        [[maybe_unused]] G4VUserTrackInformation* transferred
            = iter->second.release();
        user_info_.erase(iter);
    }
    lent_.clear();
    user_info_.clear();
}

//---------------------------------------------------------------------------//
/*!
 * At the start of an event, reset the primary ID counter.
 *
 * \pre The track data *must* have been previously flushed with the \c clear
 * command.
 */
void GeantTrackReconstruction::init_event()
{
    CELER_EXPECT(g4_track_data_.empty());
    start_ = PrimaryId(0);
    if constexpr (CELERITAS_DEBUG)
    {
        g4_event_id_ = get_current_event_id();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Register mapping from Celeritas PrimaryID to Geant4 TrackID.
 *
 * This will take ownership of the G4VUserTrackInformation and unset it in the
 * primary track. If the track was previously handed back to Geant4 with lent
 * user information, the lent information is reclaimed.
 */
PrimaryId GeantTrackReconstruction::acquire(G4Track& primary)
{
    if constexpr (CELERITAS_DEBUG)
    {
        int cur_event_id = get_current_event_id();
        CELER_VALIDATE(g4_event_id_ == cur_event_id,
                       << "GeantTrackReconstruction::init_event was not "
                          "called: last event "
                       << g4_event_id_ << " != current event " << cur_event_id);
    }
    AcquiredData data;
    data.track_id = primary.GetTrackID();
    data.parent_id = primary.GetParentID();
    data.creator_process = primary.GetCreatorProcess();
    CELER_ASSERT(data);

    // Reclaim user information lent to this track, if any
    if (auto iter = lent_.find(&primary); iter != lent_.end())
    {
        lent_.erase(iter);
    }
    if (G4VUserTrackInformation* info = primary.GetUserInformation())
    {
        if (user_info_.find(info) == user_info_.end())
        {
            // Take ownership
            user_info_.emplace(info, UPUserInfo{info});
        }
        data.user_info = info;
        // Clear user information so that it doesn't get deleted with the
        // G4Track
        primary.SetUserInformation(nullptr);
    }

    auto primary_id = start_ + g4_track_data_.size();
    g4_track_data_.push_back(data);
    return primary_id;
}

//---------------------------------------------------------------------------//
/*!
 * Restore the G4Track from the reconstruction data.
 *
 * Returns the track for the given particle ID with restored primary track
 * information.
 */
G4Track& GeantTrackReconstruction::view(ParticleId particle_id,
                                        PrimaryId primary_id) const
{
    G4Track& track = this->view(particle_id);
    this->acquired(primary_id).restore(track);
    return track;
}

//---------------------------------------------------------------------------//
/*!
 * View a track with the given particle ID.
 */
G4Track& GeantTrackReconstruction::view(ParticleId particle_id) const
{
    CELER_EXPECT(particle_id < tracks_.size());
    G4Track& track = *tracks_[particle_id.unchecked_get()];
    step_->SetTrack(&track);
    return track;
}

//---------------------------------------------------------------------------//
/*!
 * Create a new track to hand back to Geant4.
 *
 * The track has the particle type of the Celeritas track, and its identity
 * depends on its origin:
 * - \c TrackOrigin::offloaded : the Celeritas track \em is the offloaded
 *   Geant4 track, so the original track ID, parent ID, and creator process are
 *   restored, and its user information is lent to the new track.
 * - \c TrackOrigin::secondary : the track was created in Celeritas and has no
 *   Geant4 identity. Its track ID is zero (unassigned: Geant4 assigns a new
 *   one when it is stacked), its parent is the offloaded ancestor track, and
 *   its creator process is that of the ancestor. It has no user information.
 *
 * The kinematic state of the track (position, direction, energy, time,
 * etc.) must be set by the caller. The returned pointer's deleter releases
 * lent user information if the track is never handed to Geant4; once it is
 * (via \c release() on the pointer), \c release(G4Track&) \b must be called
 * before Geant4 deletes it.
 *
 * \note This must be called on the thread that will own the track because of
 * Geant4 thread-local allocators.
 */
auto GeantTrackReconstruction::create(ParticleId particle_id,
                                      PrimaryId primary_id,
                                      TrackOrigin origin) const -> UPTrack
{
    CELER_EXPECT(particle_id < tracks_.size());
    auto const& data = this->acquired(primary_id);

    G4ParticleDefinition const* pd
        = tracks_[particle_id.unchecked_get()]->GetParticleDefinition();
    CELER_ASSERT(pd);
    UPTrack result{
        new G4Track(new G4DynamicParticle(pd, G4ThreeVector()), 0.0, {}),
        TrackDeleter{this}};
    result->SetCreatorProcess(data.creator_process);

    switch (origin)
    {
        case TrackOrigin::offloaded:
            result->SetTrackID(data.track_id);
            result->SetParentID(data.parent_id);
            if (data.user_info)
            {
                CELER_ASSERT(user_info_.count(data.user_info));
                result->SetUserInformation(data.user_info);
                lent_.emplace(result.get(), data.user_info);
            }
            break;
        case TrackOrigin::secondary:
            result->SetTrackID(0);
            result->SetParentID(data.track_id);
            break;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Detach lent user information before Geant4 deletes a track.
 *
 * If the track still points to the user information lent by \c create, the
 * pointer is cleared so that deleting the track does not delete it. If user
 * code replaced the user information, the track keeps (and will delete) its
 * own. Calling this for a track without lent information has no effect.
 */
void GeantTrackReconstruction::release(G4Track& track) const
{
    auto iter = lent_.find(&track);
    if (iter == lent_.end())
    {
        return;
    }
    if (track.GetUserInformation() == iter->second)
    {
        track.SetUserInformation(nullptr);
    }
    lent_.erase(iter);
}

//---------------------------------------------------------------------------//
/*!
 * Get the acquired data for a primary in the current event.
 */
auto GeantTrackReconstruction::acquired(PrimaryId primary_id) const
    -> AcquiredData const&
{
    CELER_EXPECT(primary_id && primary_id >= start_);
    CELER_EXPECT(primary_id < start_ + g4_track_data_.size());
    if constexpr (CELERITAS_DEBUG)
    {
        int cur_event_id = get_current_event_id();
        CELER_VALIDATE(g4_event_id_ == cur_event_id,
                       << "cannot view a track from another event: "
                       << g4_event_id_ << " != current event " << cur_event_id);
    }
    return g4_track_data_[primary_id - start_];
}

//---------------------------------------------------------------------------//
// GEANTTRACKRECONSTRUCTION::ACQUIREDDATA
//---------------------------------------------------------------------------//
/*!
 * Restore the G4Track from the reconstruction data. The restored track does
 * not have ownership of the user information, user must take care to reset it
 * before deletion of the track.
 */
void GeantTrackReconstruction::AcquiredData::restore(G4Track& track) const
{
    CELER_EXPECT(*this);
    track.SetTrackID(track_id);
    track.SetParentID(parent_id);
    track.SetUserInformation(user_info);
    track.SetCreatorProcess(creator_process);
}

//---------------------------------------------------------------------------//
// GEANTTRACKRECONSTRUCTION::TRACKDELETER
//---------------------------------------------------------------------------//
/*!
 * Delete a created track after releasing any lent user information.
 */
void GeantTrackReconstruction::TrackDeleter::operator()(G4Track* track) const
{
    if (recon_ && track)
    {
        recon_->release(*track);
    }
    delete track;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
