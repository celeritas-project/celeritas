//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.cc
//---------------------------------------------------------------------------//
#include "GeantTrackReconstruction.hh"

#include <G4DynamicParticle.hh>
#include <G4ParticleDefinition.hh>
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
 * Restore the G4Track from the reconstruction data. Takes ownership of the
 * user information by unsetting it in the original track.
 */
GeantTrackReconstruction::AcquiredData::AcquiredData(G4Track& track)
    : track_id_{track.GetTrackID()}
    , parent_id_{track.GetParentID()}
    , user_info_{track.GetUserInformation()}
    , creator_process_{track.GetCreatorProcess()}
{
    CELER_EXPECT(*this);
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
PrimaryId GeantTrackReconstruction::acquire(G4Track& primary)
{
    auto primary_id = id_cast<PrimaryId>(g4_track_data_.size());
    g4_track_data_.emplace_back(AcquiredData{primary});
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
        CELER_ASSERT(primary_id < g4_track_data_.size());
        g4_track_data_[primary_id.unchecked_get()].restore(track);
    }
    return track;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
