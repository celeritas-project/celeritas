//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.hh
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
//---------------------------------------------------------------------------//
/*!
 * Manage track information for reconstruction.
 *
 * This class handles the bookkeeping of Geant4 track information needed
 * to reconstruct tracks during hit processing. It maintains mappings between
 * Celeritas PrimaryID and Geant4 track data.
 */
class GeantTrackReconstruction
{
  public:
    //!@{
    //! \name Type aliases
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    using SPStep = std::shared_ptr<G4Step>;
    //!@}

  public:
    // Construct with particle definitions for track reconstruction
    GeantTrackReconstruction(VecParticle const&, SPStep);

    ~GeantTrackReconstruction();
    CELER_DEFAULT_MOVE_DELETE_COPY(GeantTrackReconstruction);

    // Clear G4Track reconstruction data
    void clear();

    // Register mapping from Celeritas PrimaryID to Geant4 track ID
    [[nodiscard]] PrimaryId acquire(G4Track&);

    // Restore track information for given primary and particle IDs
    [[nodiscard]] G4Track& view(ParticleId, PrimaryId) const;

  private:
    //! Data needed to reconstruct a G4Track from Celeritas transport
    class AcquiredData
    {
      public:
        //! Save the G4Track reconstruction data
        explicit AcquiredData(G4Track&);
        //! Whether the data is valid
        explicit operator bool() const { return track_id_ >= 0; }
        //! Restore the G4Track from the reconstruction data
        void restore(G4Track&) const;

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
    std::vector<AcquiredData> g4_track_data_;
    //! Tracks for each particle type
    std::vector<std::unique_ptr<G4Track>> tracks_;
    //! Shared step object
    SPStep step_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
