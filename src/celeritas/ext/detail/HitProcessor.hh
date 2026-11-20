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

#include "StepProcessor.hh"
#include "TouchableUpdaterInterface.hh"

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
class HitProcessor;

//! Base class providing the host/device step pipeline for hits
using HitProcessorBase = StepProcessor<HitProcessor,
                                       DetectorStepOutput,
                                       HostRef<StepStateData>,
                                       DeviceRef<StepStateData>>;

//---------------------------------------------------------------------------//
/*!
 * Transfer Celeritas sensitive detector hits to Geant4.
 *
 * This serves a similar purpose to the \c G4FastSimHitMaker class for
 * generating hit objects.
 *
 * The asynchronous host/device pipeline (\c reserve , \c operator() ,
 * \c copy_pending_steps , \c process_copied_steps[_if_ready] ,
 * \c has_pending_steps , \c exchange_hits ) is inherited from
 * \c StepProcessor ; see that class for the state machine and its
 * constraints.
 *
 * \warning This class \b must be thread-local because the sensitive
 * detectors it points to are thread-local objects. Furthermore, Geant4
 * thread-local object allocators for the navigation state and tracks mean this
 * class \b must be destroyed on the same thread on which it was created.
 *
 * Processing a step:
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
class HitProcessor : public HitProcessorBase
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using StepPointBool = EnumArray<StepPoint, bool>;
    //!@}

    //! Label for profiling and tracing
    static constexpr char const* process_label = "process-hits";

  public:
    // Construct from volumes that have SDs and step selection
    HitProcessor(SPConstVecLV detector_volumes,
                 VecParticle const& particles,
                 StepSelection const& selection,
                 StepPointBool const& locate_touchable);
    CELER_DEFAULT_MOVE_DELETE_COPY(HitProcessor);

    // Expose base-class call operators (host, device, and full output)
    using HitProcessorBase::operator();

    // Generate and call hits from a single detector hit
    void operator()(DetectorStepOutput const& out, size_type i) const;

    // Access detector volume corresponding to an ID
    inline G4LogicalVolume const* detector_volume(DetectorId) const;

    // Access thread-local SD corresponding to an ID
    inline G4VSensitiveDetector* detector(DetectorId) const;

  private:
    //! Detector volumes for navigation updating
    SPConstVecLV detector_volumes_;
    StepSelection ss_;
    //! Map detector IDs to sensitive detectors
    std::vector<G4VSensitiveDetector*> detectors_;

    //! Step points
    EnumArray<StepPoint, G4StepPoint*> step_points_{{nullptr, nullptr}};

    //! Geant4 reference-counted pointer to a G4VTouchable
    EnumArray<StepPoint, G4TouchableHandle> touch_handle_;
    //! Navigator for finding points
    std::unique_ptr<TouchableUpdaterInterface> update_touchable_;
    //! Whether geometry-related step status can be updated
    bool step_post_status_{false};
};

// Instantiated in HitProcessor.cc
extern template class StepProcessor<HitProcessor,
                                    DetectorStepOutput,
                                    HostRef<StepStateData>,
                                    DeviceRef<StepStateData>>;

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
}  // namespace detail
}  // namespace celeritas
