//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantTrackReconstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <unordered_map>
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
//! Origin of a track reconstructed from Celeritas
enum class TrackOrigin
{
    offloaded,  //!< Same track as the one offloaded from Geant4
    secondary,  //!< Track created by Celeritas
};

//---------------------------------------------------------------------------//
/*!
 * Manage track information for reconstruction.
 *
 * This class handles the bookkeeping of Geant4 track information needed
 * to reconstruct tracks during hit processing and when handing tracks back to
 * Geant4. It maintains mappings between Celeritas PrimaryID and Geant4 track
 * data.
 *
 * \par Usage
 * - \c init_event
 * - \c acquire (multiple times)
 * - \c view (may be interleaved with acquire) to process hits
 * - \c create (may be interleaved with acquire) to hand tracks back
 * - \c clear (once all active tracks are used up)
 * - then it can be initialized with a new event, or new primaries can be
 *   added to the current event.
 *
 * \par User information
 * The user information of an offloaded track is owned by this class from \c
 * acquire until \c clear, since hits from any of its Celeritas descendants
 * may refer to it. When the offloaded track itself is handed back to Geant4
 * (\c TrackOrigin::offloaded), the same user information object is \em lent
 * to the new track. Before Geant4 deletes that track, \c release \b must be
 * called to detach the lent object. If the track is offloaded again, \c
 * acquire recognizes the lent object and resumes sole ownership. At \c clear,
 * objects still lent are transferred to the tracks that hold them, since no
 * Celeritas track can refer to them any longer.
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

    //! Delete a created track after releasing any lent user information
    class TrackDeleter
    {
      public:
        TrackDeleter() = default;
        explicit TrackDeleter(GeantTrackReconstruction const* recon)
            : recon_{recon}
        {
        }
        void operator()(G4Track* track) const;

      private:
        GeantTrackReconstruction const* recon_{nullptr};
    };

    //! Owned track created for handing back to Geant4
    using UPTrack = std::unique_ptr<G4Track, TrackDeleter>;

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

    // Restore track information for given primary and particle IDs
    [[nodiscard]] G4Track& view(ParticleId, PrimaryId) const;

    // View a track with the given particle ID
    [[nodiscard]] G4Track& view(ParticleId) const;

    // Create a new track to hand back to Geant4
    [[nodiscard]] UPTrack create(ParticleId, PrimaryId, TrackOrigin) const;

    // Detach lent user information before Geant4 deletes a track
    void release(G4Track&) const;

    // Forget user information deleted by Geant4 along with a lent track
    void forfeit(G4Track const*);

    //! Number of user information objects lent to handed-back tracks
    std::size_t num_lent() const { return lent_.size(); }

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
        //! User track information (owned by the reconstruction)
        G4VUserTrackInformation* user_info{nullptr};
        //! Process that created the track
        G4VProcess const* creator_process{nullptr};

        //! Whether the data is valid
        explicit operator bool() const { return track_id >= 0; }
        //! Restore the G4Track from the reconstruction data
        void restore(G4Track&) const;
    };

    using UPUserInfo = std::unique_ptr<G4VUserTrackInformation>;

    //! G4Track reconstruction data indexed by Celeritas PrimaryID
    std::vector<AcquiredData> g4_track_data_;
    //! Tracks for each particle type
    std::vector<std::unique_ptr<G4Track>> tracks_;
    //! Shared step object
    SPStep step_;
    //! Starting primary id
    PrimaryId start_{0};
    //! Last G4 event ID for error checking
    int g4_event_id_{-1};

    //! User information owned by this class
    std::unordered_map<G4VUserTrackInformation const*, UPUserInfo> user_info_;
    //! User information lent to tracks handed back to Geant4
    mutable std::unordered_map<G4Track const*, G4VUserTrackInformation*> lent_;

    // Get acquired data for a primary
    AcquiredData const& acquired(PrimaryId) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
