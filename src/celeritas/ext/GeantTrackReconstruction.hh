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
 * A Celeritas primary (generation 0) \em is the Geant4 track that was
 * offloaded: it is reconstructed with the Geant4 track ID, parent ID, creator
 * process, and user information saved by \c acquire , whatever their values.
 * A track created by Celeritas has its own Geant4 identity, which never
 * refers to the information of its offloaded ancestor:
 * - its track ID is \c geant_track_id of its Celeritas track ID,
 * - its parent ID is the saved Geant4 ID of the offloaded track if it is in
 *   generation 1 (see \c SimTrackView::generation ), and otherwise the
 *   mapped ID of its Celeritas parent,
 * - its creator process is an inert placeholder named "celeritas", and
 * - it has no user information.
 *
 * A track that Geant4 offloads again after Celeritas handed it back is a new
 * primary: it keeps the Geant4 ID it had, and its secondaries refer to it.
 *
 * \note Tracks created by Celeritas are never seen by Geant4: no user
 * tracking action is called for them and they have no trajectory. User code
 * that looks up information by track ID (e.g., recorded in a user tracking
 * action) will not find them or, beyond generation 1, their parents, and
 * sensitive detectors must not assume that user track information is set.
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
    CELER_DEFAULT_MOVE_DELETE_COPY(GeantTrackReconstruction);

    // Clear G4Track reconstruction data
    void clear();

    // Register mapping from Celeritas PrimaryID to Geant4 track ID
    [[nodiscard]] PrimaryId acquire(G4Track&);

    // Reset primary ID at each event start
    void init_event();

    // Geant4 track ID of a track created by Celeritas
    static int geant_track_id(TrackId);

    // Restore the Geant4 identity of any Celeritas track
    [[nodiscard]] G4Track& view(ParticleId particle,
                                PrimaryId primary,
                                TrackId track,
                                TrackId parent,
                                size_type generation) const;

    // View a track with the given particle ID
    [[nodiscard]] G4Track& view(ParticleId) const;

    //! Creator process reported for tracks created by Celeritas
    G4VProcess const& placeholder_process() const { return *placeholder_; }

    // Event ID function pointer for unit testing (only used in
    // CELERITAS_DEBUG)
    static EventIdGetter get_current_event_id;

  private:
    //! Data needed to reconstruct a G4Track from Celeritas transport
    class AcquiredData
    {
      public:
        //! Save the G4Track reconstruction data
        explicit AcquiredData(G4Track&);
        //! Whether the data is valid
        explicit operator bool() const { return track_id_ >= 0; }
        //! Original Geant4 track ID
        int track_id() const { return track_id_; }
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
    //! Creator process of Celeritas tracks (owned by G4ProcessTable)
    G4VProcess* placeholder_{nullptr};
    //! Starting primary id
    PrimaryId start_{0};
    //! Last G4 event ID for error checking
    int g4_event_id_{-1};

    // Get acquired data for a primary
    AcquiredData const& acquired(PrimaryId) const;

    // Restore the saved identity of an offloaded track
    G4Track& view(ParticleId, PrimaryId) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
