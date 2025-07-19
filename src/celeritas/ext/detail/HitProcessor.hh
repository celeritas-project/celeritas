//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HitProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <G4TouchableHandle.hh>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "TouchableUpdaterInterface.hh"

class G4LogicalVolume;
class G4ParticleDefinition;
class G4Step;
class G4StepPoint;
class G4Track;
class G4VProcess;
class G4VSensitiveDetector;
class G4VUserTrackInformation;

namespace celeritas
{
struct StepSelection;
struct DetectorStepOutput;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Transfer Celeritas sensitive detector hits to Geant4.
 *
 * This serves a similar purpose to the \c G4FastSimHitMaker class for
 * generating hit objects.
 *
 * \warning This class \b must be thread-local because the sensitive
 * detectors it points to are thread-local objects. Furthermore, Geant4
 * thread-local object allocators for the navigation state and tracks mean this
 * class \b must be destroyed on the same thread on which it was created.
 *
 * Call operator:
 * - Loop over detector steps
 * - Update step attributes based on hit selection for the detector (TODO:
 *   selection is global for now)
 * - Call the local detector (based on detector ID from map) with the step
 *
 * Compare to Geant4 updating step/track info:
 * - \c G4VParticleChange::UpdateStepInfo
 * - \c G4ParticleChangeForTransport::UpdateStepForAlongStep
 * - \c G4ParticleChangeForTransport::UpdateStepForPostStep
 * - \c G4StackManager::PrepareNewEvent
 * - \c G4SteppingManager::ProcessSecondariesFromParticleChange
 * - \c G4Step::UpdateTrack
 */
class HitProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    using StepPointBool = EnumArray<StepPoint, bool>;
    //!@}

  public:
    // Construct from volumes that have SDs and step selection
    HitProcessor(SPConstVecLV detector_volumes,
                 SPConstCoreGeo const& geo,
                 VecParticle const& particles,
                 StepSelection const& selection,
                 StepPointBool const& locate_touchable);

    // Log on destruction
    ~HitProcessor();
    CELER_DEFAULT_MOVE_DELETE_COPY(HitProcessor);

    // Process CPU-generated hits
    void operator()(StepStateHostRef const&);

    // Process device-generated hits
    void operator()(StepStateDeviceRef const&);

    // Generate and call hits from a detector output (for testing)
    void operator()(DetectorStepOutput const& out) const;

    // Generate and call hits from a single detector hit
    void operator()(DetectorStepOutput const& out, size_type i) const;

    // Access detector volume corresponding to an ID
    inline G4LogicalVolume const* detector_volume(DetectorId) const;

    // Access thread-local SD corresponding to an ID
    inline G4VSensitiveDetector* detector(DetectorId) const;

    // Get and reset the hits counted (generally once per event)
    inline size_type exchange_hits();

    // Register mapping from Celeritas PrimaryID to Geant4 TrackID
    [[nodiscard]] PrimaryId register_primary(G4Track&);

    // Clear G4Track reconstruction data
    void end_event();

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
        //! User track information
        std::unique_ptr<G4VUserTrackInformation> user_info_;
        //! Process that created the track
        G4VProcess const* creator_process_{nullptr};
    };

    //! Detector volumes for navigation updating
    SPConstVecLV detector_volumes_;
    //! Map detector IDs to sensitive detectors
    std::vector<G4VSensitiveDetector*> detectors_;
    //! Temporary CPU hit information
    DetectorStepOutput steps_;

    //! Temporary step
    std::unique_ptr<G4Step> step_;
    //! Step points
    EnumArray<StepPoint, G4StepPoint*> step_points_{{nullptr, nullptr}};
    //! Tracks for each particle type
    std::vector<std::unique_ptr<G4Track>> tracks_;

    //! Geant4 reference-counted pointer to a G4VTouchable
    EnumArray<StepPoint, G4TouchableHandle> touch_handle_;
    //! Navigator for finding points
    std::unique_ptr<TouchableUpdaterInterface> update_touchable_;
    //! Whether geometry-related step status can be updated
    bool step_post_status_{false};

    //! Accumulated number of hits
    size_type num_hits_;

    //! G4Track reconstruction data indexed by Celeritas PrimaryID
    std::vector<GeantTrackReconstructionData> g4_track_data_;

    void update_track(DetectorStepOutput const& out, size_type i) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access detector volume corresponding to an ID.
 */
G4LogicalVolume const* HitProcessor::detector_volume(DetectorId did) const
{
    CELER_EXPECT(did < detector_volumes_->size());
    return (*detector_volumes_)[did.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access thread-local sensitive detector corresponding to an ID.
 */
G4VSensitiveDetector* HitProcessor::detector(DetectorId did) const
{
    CELER_EXPECT(did < detectors_.size());
    return detectors_[did.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get and reset number of hits counted (generally once per event).
 */
size_type HitProcessor::exchange_hits()
{
    return std::exchange(num_hits_, size_type{0});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
