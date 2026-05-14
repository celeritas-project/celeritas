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
#include <G4Navigator.hh>
#include <G4ParticleDefinition.hh>
#include <G4PrimaryParticle.hh>
#include <G4Step.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4TransportationManager.hh>
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

//---------------------------------------------------------------------------//
void update_touchable(G4Track& track)
{
    auto* navigator = G4TransportationManager::GetTransportationManager()
                          ->GetNavigatorForTracking();
    CELER_ASSERT(navigator);

    auto const& pos = track.GetPosition();
    auto const& dir = track.GetMomentumDirection();
    auto* volume = navigator->LocateGlobalPointAndSetup(
        pos, &dir, /*pRelativeSearch=*/false, /*ignoreDirection=*/false);
    CELER_VALIDATE(volume,
                   << "failed to locate reconstructed Geant4 track "
                      "at position "
                   << pos << " in the tracking geometry");
    track.SetTouchableHandle(navigator->CreateTouchableHistoryHandle());
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
GeantTrackReconstruction::GeantTrackReconstruction(VecParticle const& particles,
                                                   SPStep step)
    : step_(std::move(step))
{
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
 * an event is complete.
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
 * Register mapping from Celeritas PrimaryID to Geant4 TrackID. This will take
 * ownership of the G4VUserTrackInformation and unset it in the primary track.
 */
PrimaryId
GeantTrackReconstruction::acquire(G4Track& primary, ParticleId particle_id)
{
    if constexpr (CELERITAS_DEBUG)
    {
        int cur_event_id = get_current_event_id();
        CELER_VALIDATE(g4_event_id_ == cur_event_id,
                       << "GeantTrackReconstruction::init_event was not "
                          "called: last event "
                       << g4_event_id_ << " != current event " << cur_event_id);
    }
    auto primary_id = start_ + g4_track_data_.size();
    g4_track_data_.emplace_back(AcquiredData{primary, particle_id});
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
    CELER_EXPECT(primary_id && primary_id >= start_);
    CELER_EXPECT(primary_id < start_ + g4_track_data_.size());
    if constexpr (CELERITAS_DEBUG)
    {
        int cur_event_id = get_current_event_id();
        CELER_VALIDATE(g4_event_id_ == cur_event_id,
                       << "cannot view a track from another event: "
                       << g4_event_id_ << " != current event " << cur_event_id);
    }

    G4Track& track = this->view(particle_id);
    g4_track_data_[this->local_index(primary_id)].restore(track);
    return track;
}

//---------------------------------------------------------------------------//
/*!
 * Restore the track with its initial handover state.
 */
G4Track& GeantTrackReconstruction::view_initial(ParticleId particle_id,
                                                PrimaryId primary_id) const
{
    CELER_EXPECT(particle_id < tracks_.size());
    CELER_EXPECT(primary_id && primary_id >= start_);
    CELER_EXPECT(primary_id < start_ + g4_track_data_.size());
    if constexpr (CELERITAS_DEBUG)
    {
        int cur_event_id = get_current_event_id();
        CELER_VALIDATE(g4_event_id_ == cur_event_id,
                       << "cannot view a track from another event: "
                       << g4_event_id_ << " != current event " << cur_event_id);
    }

    G4Track& track = this->view(particle_id);
    g4_track_data_[this->local_index(primary_id)].restore_initial(track);
    return track;
}

//---------------------------------------------------------------------------//
/*!
 * Return true if the given primary was created by the event generator.
 */
bool GeantTrackReconstruction::is_generator_primary(PrimaryId primary_id) const
{
    return g4_track_data_[this->local_index(primary_id)].is_generator_primary();
}

//---------------------------------------------------------------------------//
/*!
 * Get the Celeritas particle type for a given primary.
 */
ParticleId GeantTrackReconstruction::particle_id(PrimaryId primary_id) const
{
    return g4_track_data_[this->local_index(primary_id)].particle_id();
}

//---------------------------------------------------------------------------//
/*!
 * Convert an event-scoped primary ID to the current reconstruction index.
 */
size_type GeantTrackReconstruction::local_index(PrimaryId primary_id) const
{
    CELER_EXPECT(primary_id && primary_id >= start_);
    CELER_EXPECT(primary_id < start_ + g4_track_data_.size());
    return primary_id.unchecked_get() - start_.unchecked_get();
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
// GEANTTRACKRECONSTRUCTION::ACQUIREDDATA
//---------------------------------------------------------------------------//
/*!
 * Save G4Track reconstruction data along with the Celeritas particle type.
 */
GeantTrackReconstruction::AcquiredData::AcquiredData(G4Track& track,
                                                     ParticleId particle_id)
    : track_id_{track.GetTrackID()}
    , parent_id_{track.GetParentID()}
    , particle_id_{particle_id}
    , kinetic_energy_{track.GetKineticEnergy()}
    , time_{track.GetGlobalTime()}
    , primary_particle_{track.GetDynamicParticle()->GetPrimaryParticle()}
    , user_info_{track.GetUserInformation()}
    , creator_process_{track.GetCreatorProcess()}
{
    CELER_EXPECT(*this);
    auto const& pos = track.GetPosition();
    pos_[0] = pos.x();
    pos_[1] = pos.y();
    pos_[2] = pos.z();
    auto const& dir = track.GetMomentumDirection();
    dir_[0] = dir.x();
    dir_[1] = dir.y();
    dir_[2] = dir.z();
    // Clear user information so that it doesn't get deleted with the G4Track
    track.SetUserInformation(nullptr);
}

//---------------------------------------------------------------------------//
/*!
 * Restore the G4Track from the reconstruction data. The restored track does
 * not have ownership of the user information, user must take care to reset it
 * before deletion of the track.
 */
void GeantTrackReconstruction::AcquiredData::restore(G4Track& track) const
{
    CELER_EXPECT(*this);
    track.SetTrackID(track_id_);
    track.SetParentID(parent_id_);
    track.SetUserInformation(user_info_.get());
    track.SetCreatorProcess(creator_process_);
}

//---------------------------------------------------------------------------//
/*!
 * Restore the initial kinematic state for tracking-action dispatch.
 */
void GeantTrackReconstruction::AcquiredData::restore_initial(G4Track& track) const
{
    CELER_EXPECT(*this);
    this->restore(track);
    track.SetPosition(G4ThreeVector(pos_[0], pos_[1], pos_[2]));
    track.SetMomentumDirection(G4ThreeVector(dir_[0], dir_[1], dir_[2]));
    track.SetGlobalTime(time_);
    auto* dp = const_cast<G4DynamicParticle*>(track.GetDynamicParticle());
    dp->SetKineticEnergy(kinetic_energy_);
    dp->SetPrimaryParticle(const_cast<G4PrimaryParticle*>(primary_particle_));
    update_touchable(track);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
