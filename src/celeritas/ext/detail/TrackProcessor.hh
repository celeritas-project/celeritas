//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TrackProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

class G4ParticleDefinition;
class G4Step;
class G4Track;
class G4VProcess;
class G4VUserTrackInformation;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage track information for reconstruction.
 *
 * This class handles the bookkeeping of Geant4 track information needed
 * to reconstruct tracks during hit processing. It maintains mappings between
 * Celeritas PrimaryID and Geant4 track data.
 *
 */
class TrackProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    //!@}

  public:
    // Construct with particle definitions for track reconstruction
    explicit TrackProcessor(VecParticle const&);

    ~TrackProcessor();
    CELER_DEFAULT_MOVE_DELETE_COPY(TrackProcessor);

    // Clear G4Track reconstruction data
    void end_event();

    // Register mapping from Celeritas PrimaryID to Geant4 TrackID
    [[nodiscard]] PrimaryId register_primary(G4Track&);

    // Restore track information for given primary and particle IDs
    G4Track& restore_track(ParticleId, PrimaryId) const;

    // Get the owned step
    G4Step& step() { return *step_; }

  private:
    //! Data needed to reconstruct a G4Track from Celeritas transport
    class GeantTrackReconstructionData
    {
      public:
        //! Save the G4Track reconstruction data
        explicit GeantTrackReconstructionData(G4Track&);
        //! Whether the data is valid
        explicit operator bool() const { return track_id_ >= 0; }
        //! Restore the G4Track from the reconstruction data
        void restore_track(G4Track&) const;

      private:
        //! Original Geant4 track ID
        int track_id_{-1};
        //! Original Geant4 parent ID
        int parent_id_{0};
        //! User track information
        std::unique_ptr<G4VUserTrackInformation> user_info_;
        //! Process that created the track
        G4VProcess const* creator_process_{nullptr};
    };

    //! G4Track reconstruction data indexed by Celeritas PrimaryID
    std::vector<GeantTrackReconstructionData> g4_track_data_;
    //! Tracks for each particle type
    std::vector<std::unique_ptr<G4Track>> tracks_;
    //! Owned step object
    std::unique_ptr<G4Step> step_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
