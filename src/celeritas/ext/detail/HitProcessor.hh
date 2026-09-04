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
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "TouchableUpdaterInterface.hh"
#include "../GeantTrackReconstruction.hh"

class G4LogicalVolume;
class G4ParticleDefinition;
class G4Step;
class G4StepPoint;
class G4Track;
class G4VSensitiveDetector;

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
 * Host step data is copied and processed immediately by the call operator.
 * For device step data, the call operator retains a reference to the gathered
 * step state without copying it. After the producing step is complete, the
 * caller must call \c process_pending_steps before launching another step that
 * can overwrite the shared step state. Only one device step can be pending.
 * Processing currently copies the selected detector data synchronously to
 * pinned host storage, then:
 * - loops over detector steps;
 * - updates step attributes based on the hit selection (TODO: selection is
 *   global for now); and
 * - calls the local detector selected by detector ID with the step.
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
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    using StepPointBool = EnumArray<StepPoint, bool>;
    //!@}

  public:
    // Construct from volumes that have SDs and step selection
    HitProcessor(SPConstVecLV detector_volumes,
                 VecParticle const& particles,
                 StepSelection const& selection,
                 StepPointBool const& locate_touchable);

    ~HitProcessor() = default;
    CELER_DEFAULT_MOVE_DELETE_COPY(HitProcessor);

    // Process CPU-generated hits
    void operator()(StepStateHostRef const&);

    // Save device-generated hits for processing after step completion
    void operator()(StepStateDeviceRef const&);

    // Copy and process device-generated hits after their step completes
    void process_pending_steps();

    //! Whether device-generated hit data is pending
    bool has_pending_steps() const noexcept
    {
        return static_cast<bool>(pending_device_steps_);
    }

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

    //! Access local Geant4 track metadata reconstruction
    std::shared_ptr<GeantTrackReconstruction> const& track_reconstruction() const
    {
        return track_reconstruction_;
    }

  private:
    //! Detector volumes for navigation updating
    SPConstVecLV detector_volumes_;
    StepSelection ss_;
    //! Map detector IDs to sensitive detectors
    std::vector<G4VSensitiveDetector*> detectors_;
    //! Temporary CPU hit information
    DetectorStepOutput steps_;

    //! Device step data awaiting transfer after step completion
    StepStateDeviceRef pending_device_steps_;

    //! Shared step object
    std::shared_ptr<G4Step> step_;

    //! Track reconstruction for hit processing
    std::shared_ptr<GeantTrackReconstruction> track_reconstruction_;
    //! Step points
    EnumArray<StepPoint, G4StepPoint*> step_points_{{nullptr, nullptr}};

    //! Geant4 reference-counted pointer to a G4VTouchable
    EnumArray<StepPoint, G4TouchableHandle> touch_handle_;
    //! Navigator for finding points
    std::unique_ptr<TouchableUpdaterInterface> update_touchable_;
    //! Whether geometry-related step status can be updated
    bool step_post_status_{false};

    //! Accumulated number of hits
    size_type num_hits_{0};

    // Process hits already copied into temporary CPU storage
    void process_local_steps();
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
    using namespace celeritas::literals;
    CELER_EXPECT(!this->has_pending_steps());
    return std::exchange(num_hits_, 0_sz);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
