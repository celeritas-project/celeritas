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
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/sys/DeviceEvent.hh"
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
 * Device step data goes through three phases so that the device-to-host
 * transfer and the Geant4 hit processing can overlap with the next step:
 * 1. \b Compacted: the call operator enqueues the selection and compaction of
 *    detector steps into the state's scratch space, along with an
 *    asynchronous copy of the number of selected steps to pinned host memory,
 *    without synchronizing. Because the enqueued kernels read the gathered
 *    step data, it may be overwritten only by work enqueued later on the same
 *    stream.
 * 2. \b Copied: once the producing step is complete, \c copy_pending_steps
 *    reads the number of selected steps and enqueues the copy of the
 *    compacted data to pinned host storage on the device's \c memcpy_stream ,
 *    without synchronizing. The next compaction makes the state's stream wait
 *    for this copy, so the next step can be launched immediately.
 * 3. \b Processed: \c process_copied_steps waits for the copy, or
 *    \c process_copied_steps_if_ready polls it, and then reconstructs hits.
 *
 * These phases move each step through two buffers, the device \em scratch
 * space and the pinned host \em output, which act as a two-stage pipeline:
 * step \em n+1 can be compacted in the scratch while step \em n is still
 * copied in the output, waiting to be processed.
 * \verbatim
                      operator()(DeviceRef)
   scratch:   free ------------------------> compacted
               ^                                 |
               +------- copy_pending_steps ------+
                                                 | (enqueues the copy)
                                                 v
   output:    empty <----------------------- copied
                 process_copied_steps[_if_ready]
   \endverbatim
 *
 * When the scratch is "free", the device may still be reading it for the
 * previous copy: the next compaction waits for that copy on the device. The
 * calls allowed in each combined (scratch, output) state are:
 * - (free, empty): \c reserve , \c operator() and \c exchange_hits ;
 * - (compacted, empty): \c copy_pending_steps ;
 * - (free, copied): \c operator() and \c process_copied_steps[_if_ready] ;
 * - (compacted, copied): \c copy_pending_steps , which processes the output
 *   first, and \c process_copied_steps[_if_ready] .
 *
 * Copy and process calls in other states are no-ops, and
 * \c has_pending_steps is true in every state but (free, empty). The output
 * state is stored, while the scratch state is derived from whether compacted
 * step data is pending, so each buffer's state has a single source.
 *
 * \warning The next compaction waits only for this processor's copy. Since
 * all callbacks of a \c StepCollector share the scratch space, no other
 * callback may compact into it with \c compact_steps_async while a copy is in
 * flight, or the copied hits would be silently replaced with later step data.
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

    // Wait for any pending device compaction before releasing its buffers
    ~HitProcessor();
    CELER_DEFAULT_MOVE_DELETE_COPY(HitProcessor);

    // Process CPU-generated hits
    void operator()(StepStateHostRef const&);

    // Allocate pinned host output for the largest possible device step
    void reserve(StepStateDeviceRef const&);

    // Enqueue compaction of device-generated hits for later processing
    void operator()(StepStateDeviceRef const&);

    // Enqueue the copy of compacted device hits after their step completes
    void copy_pending_steps();

    // Wait for and process copied device hits
    void process_copied_steps();

    // Process copied device hits if the copy has completed
    void process_copied_steps_if_ready();

    //! Whether device-generated hit data is pending
    bool has_pending_steps() const noexcept
    {
        return this->scratch_state() != ScratchState::free
               || output_ != OutputState::empty;
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
    //! State of the device scratch space (see the class documentation)
    enum class ScratchState
    {
        free,  //!< Available for the next compaction
        compacted,  //!< Holds compacted steps awaiting a copy
    };

    //! State of the pinned host output (see the class documentation)
    enum class OutputState
    {
        empty,  //!< Available for the next copy
        copied,  //!< Holds copied steps awaiting processing
    };

    //! Detector volumes for navigation updating
    SPConstVecLV detector_volumes_;
    StepSelection ss_;
    //! Map detector IDs to sensitive detectors
    std::vector<G4VSensitiveDetector*> detectors_;
    //! Temporary CPU hit information
    DetectorStepOutput steps_;

    //! Device step data compacted in the scratch space, awaiting a copy
    StepStateDeviceRef pending_device_steps_;
    //! Number of compacted device steps (pinned, allocated if device is used)
    std::vector<size_type, PinnedAllocator<size_type>> num_selected_;
    //! Completion of the pending device step compaction
    DeviceEvent compacted_{nullptr};
    //! Completion of the copy of compacted steps to \c steps_
    DeviceEvent copied_{nullptr};
    //! Whether \c steps_ holds copied device steps awaiting processing
    OutputState output_{OutputState::empty};
    //! Whether \c steps_ is reserved for the largest possible device step
    bool steps_reserved_{false};

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

    //! State of the scratch space, derived from the pending step data
    ScratchState scratch_state() const noexcept
    {
        return pending_device_steps_ ? ScratchState::compacted
                                     : ScratchState::free;
    }
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
