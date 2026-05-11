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
#include "corecel/Types.hh"
#include "celeritas/Types.hh"

class G4ParticleDefinition;
class G4PrimaryParticle;
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
    [[nodiscard]] PrimaryId acquire(G4Track&, ParticleId);

    // Reset primary ID at each event start
    void init_event();

    // Restore track terminal information for given primary and particle IDs
    [[nodiscard]] G4Track& view(ParticleId, PrimaryId) const;

    // Restore track initial handover state for tracking-action dispatch
    [[nodiscard]] G4Track& view_initial(ParticleId, PrimaryId) const;

    // View a track with the given particle ID
    [[nodiscard]] G4Track& view(ParticleId) const;

    // True if the given primary was created by the event generator
    bool is_generator_primary(PrimaryId) const;

    // Iterate over acquired primaries using event-scoped PrimaryId values
    template<class F>
    void for_each_primary(F&& func) const;

    // Celeritas particle type for a given primary
    ParticleId particle_id(PrimaryId) const;

    // Event ID function pointer for unit testing (only used in
    // CELERITAS_DEBUG)
    static EventIdGetter get_current_event_id;

  private:
    //! Data needed to reconstruct a G4Track from Celeritas transport
    class AcquiredData
    {
      public:
        //! Save the G4Track reconstruction data along with ParticleId
        AcquiredData(G4Track&, ParticleId);
        //! Whether the data is valid
        explicit operator bool() const { return track_id_ >= 0; }
        //! Restore the G4Track from the reconstruction data
        void restore(G4Track&) const;
        //! Restore initial kinematic state
        void restore_initial(G4Track&) const;
        //! Celeritas particle type for this primary
        ParticleId particle_id() const { return particle_id_; }
        //! True if this track was created by the event generator
        bool is_generator_primary() const { return parent_id_ == 0; }

      private:
        //! Original Geant4 track ID
        int track_id_{-1};
        //! Original Geant4 parent ID
        int parent_id_{0};
        //! Celeritas particle type
        ParticleId particle_id_{};
        //! Initial kinetic energy [MeV]
        double kinetic_energy_{0};
        //! Initial global time [ns]
        double time_{0};
        //! Initial position [mm]
        double pos_[3]{0, 0, 0};
        //! Initial momentum direction
        double dir_[3]{0, 0, 1};
        //! Generator-level primary particle pointer
        G4PrimaryParticle const* primary_particle_{nullptr};
        //! User track information
        std::unique_ptr<G4VUserTrackInformation> user_info_;
        //! Process that created the track
        G4VProcess const* creator_process_{nullptr};
    };

    // Convert an event-scoped primary ID to the current reconstruction index
    size_type local_index(PrimaryId) const;

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
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Iterate over acquired primaries without exposing flush-local indices.
 */
template<class F>
void GeantTrackReconstruction::for_each_primary(F&& func) const
{
    for (size_type i = 0; i < g4_track_data_.size(); ++i)
    {
        PrimaryId pid = start_ + i;
        auto const& data = g4_track_data_[i];
        func(pid, data.particle_id());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
