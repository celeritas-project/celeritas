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
 *
 * \par Usage
 * - \c init_event
 * - \c acquire (multiple times)
 * - \c view (may be interleaved with acquire)
 * - \c clear (once all active tracks are used up)
 * - then it can be initialized with a new event, or new primaries can be
 *   added to the current event.
 *
 * \par Track identity
 * A Celeritas track without a parent \em is the Geant4 track that was
 * offloaded: it is reconstructed with the original track ID, parent ID,
 * creator process, and user information. A track created by Celeritas has
 * its own Geant4 identity, which never refers to the information of its
 * offloaded ancestor:
 * - its track ID is \c geant_track_id of its Celeritas track ID,
 * - its parent ID is the original ID of the offloaded track if its parent is
 *   that track, and otherwise the mapped ID of its Celeritas parent,
 * - its creator process is an inert placeholder named "celeritas", and
 * - it has no user information.
 *
 * \par User information
 * The user information of an offloaded track is owned by this class from \c
 * acquire until \c clear .
 */
class GeantTrackReconstruction
{
  public:
    //!@{
    //! \name Type aliases
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    using SPStep = std::shared_ptr<G4Step>;
    using EventIdGetter = int (*)();
    //!@}

  public:
    // Create a G4Step object with cleared data
    static SPStep make_g4step();

    // Construct with particle definitions for track reconstruction
    GeantTrackReconstruction(VecParticle const&, SPStep);

    ~GeantTrackReconstruction();
    CELER_DELETE_COPY_MOVE(GeantTrackReconstruction);

    // Clear G4Track reconstruction data
    void clear();

    // Register mapping from Celeritas PrimaryID to Geant4 track ID
    [[nodiscard]] PrimaryId acquire(G4Track&);

    // Reset primary ID at each event start
    void init_event();

    // Geant4 track ID of a track created by Celeritas
    static int geant_track_id(TrackId);

    // Restore track information for given primary and particle IDs
    [[nodiscard]] G4Track& view(ParticleId, PrimaryId) const;

    // Restore the Geant4 identity of any Celeritas track
    [[nodiscard]] G4Track& view(ParticleId particle,
                                PrimaryId primary,
                                TrackId track,
                                TrackId parent,
                                bool parent_is_primary) const;

    // View a track with the given particle ID
    [[nodiscard]] G4Track& view(ParticleId) const;

    //! Creator process reported for tracks created by Celeritas
    G4VProcess const& placeholder_process() const { return *placeholder_; }

    // Event ID function pointer for unit testing (only used in
    // CELERITAS_DEBUG)
    static EventIdGetter get_current_event_id;

  private:
    //! Data needed to reconstruct a G4Track from Celeritas transport
    struct AcquiredData
    {
        //! Original Geant4 track ID
        int track_id{-1};
        //! Original Geant4 parent ID
        int parent_id{0};
        //! User track information
        std::unique_ptr<G4VUserTrackInformation> user_info;
        //! Process that created the track
        G4VProcess const* creator_process{nullptr};

        //! Whether the data is valid
        explicit operator bool() const { return track_id >= 0; }
        //! Restore the G4Track from the reconstruction data
        void restore(G4Track&) const;
    };

    //! G4Track reconstruction data indexed by Celeritas PrimaryID
    std::vector<AcquiredData> g4_track_data_;
    //! Tracks for each particle type
    std::vector<std::unique_ptr<G4Track>> tracks_;
    //! Shared step object
    SPStep step_;
    //! Creator process of Celeritas tracks
    std::unique_ptr<G4VProcess> placeholder_;
    //! Starting primary id
    PrimaryId start_{0};
    //! Last G4 event ID for error checking
    int g4_event_id_{-1};

    // Get acquired data for a primary
    AcquiredData const& acquired(PrimaryId) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
