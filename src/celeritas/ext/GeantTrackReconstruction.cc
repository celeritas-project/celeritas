//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.cc
//---------------------------------------------------------------------------//
#include "GeantTrackReconstruction.hh"

#include <G4DynamicParticle.hh>
#include <G4ParticleDefinition.hh>
#include <G4PrimaryParticle.hh>
#include <G4Step.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4VProcess.hh>
#include <G4VUserTrackInformation.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save G4Track reconstruction data along with the Celeritas particle type.
 * Takes ownership of the user information by unsetting it in the original
 * track.
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
 * Restore the initial kinematic state for PreUserTrackingAction dispatch.
 *
 * Sets position, momentum direction, kinetic energy, global time, and the
 * G4PrimaryParticle pointer on the dynamic particle so that MC-truth
 * frameworks that check GetPrimaryParticle() in PreUserTrackingAction
 * correctly identify the track as a generator-level primary.
 */
void GeantTrackReconstruction::AcquiredData::restore_initial(G4Track& track) const
{
    CELER_EXPECT(*this);
    restore(track);
    track.SetPosition(G4ThreeVector(pos_[0], pos_[1], pos_[2]));
    track.SetMomentumDirection(G4ThreeVector(dir_[0], dir_[1], dir_[2]));
    track.SetGlobalTime(time_);
    auto* dp = const_cast<G4DynamicParticle*>(track.GetDynamicParticle());
    dp->SetKineticEnergy(kinetic_energy_);
    dp->SetPrimaryParticle(const_cast<G4PrimaryParticle*>(primary_particle_));
}

//---------------------------------------------------------------------------//
/*!
 * Return true if the given primary was created by the event generator.
 *
 * Generator-level primaries have parent ID 0. Geant4-tracked secondaries that
 * were re-offloaded to Celeritas via HandOverOneTrack have a non-zero parent
 * ID, so Pre/PostUserTrackingAction should not be fired for them here.
 */
bool GeantTrackReconstruction::is_generator_primary(PrimaryId primary_id) const
{
    CELER_EXPECT(primary_id);
    CELER_ASSERT(primary_id.unchecked_get() < g4_track_data_.size());
    return g4_track_data_[primary_id.unchecked_get()].is_generator_primary();
}

//---------------------------------------------------------------------------//
/*!
 * Get the Celeritas particle type for a given primary.
 */
ParticleId GeantTrackReconstruction::particle_id(PrimaryId primary_id) const
{
    CELER_EXPECT(primary_id);
    CELER_ASSERT(primary_id.unchecked_get() < g4_track_data_.size());
    return g4_track_data_[primary_id.unchecked_get()].particle_id();
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
        this->clear();
    }
    catch (...)  // NOLINT(bugprone-empty-catch)
    {
        // Ignore anything bad that happens while logging
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear G4Track reconstruction data.
 */
void GeantTrackReconstruction::clear()
{
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
 * Register mapping from Celeritas PrimaryID to Geant4 TrackID. This will take
 * ownership of the G4VUserTrackInformation and unset it in the primary track.
 */
PrimaryId
GeantTrackReconstruction::acquire(G4Track& primary, ParticleId particle_id)
{
    auto primary_id = celeritas::id_cast<PrimaryId>(g4_track_data_.size());
    g4_track_data_.emplace_back(AcquiredData{primary, particle_id});
    return primary_id;
}

//---------------------------------------------------------------------------//
/*!
 * Restore the G4Track from the reconstruction data. Returns the track for the
 * given particle ID with restored primary track information if a valid
 * PrimaryId is provided.
 */
G4Track& GeantTrackReconstruction::view(ParticleId particle_id,
                                        PrimaryId primary_id) const
{
    CELER_EXPECT(particle_id < tracks_.size());

    G4Track& track = *tracks_[particle_id.unchecked_get()];

    step_->SetTrack(&track);

    if (primary_id)
    {
        // primary_id is flush-local: direct index into g4_track_data_
        CELER_ASSERT(primary_id.unchecked_get() < g4_track_data_.size());
        g4_track_data_[primary_id.unchecked_get()].restore(track);
    }
    return track;
}

//---------------------------------------------------------------------------//
/*!
 * Restore the track with its initial (handover) state.
 *
 * Used to prepare the track for PreUserTrackingAction in Flush(), where the
 * track must look exactly as it did when first handed over to Celeritas.
 */
G4Track& GeantTrackReconstruction::view_initial(ParticleId particle_id,
                                                PrimaryId primary_id) const
{
    CELER_EXPECT(particle_id < tracks_.size());
    CELER_EXPECT(primary_id);

    G4Track& track = *tracks_[particle_id.unchecked_get()];
    step_->SetTrack(&track);

    // primary_id is flush-local: direct index into g4_track_data_
    CELER_ASSERT(primary_id.unchecked_get() < g4_track_data_.size());
    g4_track_data_[primary_id.unchecked_get()].restore_initial(track);

    return track;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
