//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HandBackProcessor.hh
//! \sa test/celeritas/ext/GeantHandBack.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/Types.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "TouchableUpdaterInterface.hh"
#include "../GeantTrackReconstruction.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * A track reconstructed from Celeritas for handing back to Geant4.
 */
struct HandedBackTrack
{
    using UPTrack = GeantTrackReconstruction::UPTrack;

    UPTrack track;  //!< Reconstructed Geant4 track
    TrackOrigin origin{TrackOrigin::offloaded};  //!< Geant4 identity
    HandBackReason reason{HandBackReason::none};  //!< Why it's handed back
    TrackId celer_track_id;  //!< Celeritas track ID
    TrackId celer_parent_id;  //!< Celeritas parent track ID
};

//---------------------------------------------------------------------------//
/*!
 * Reconstruct Geant4 tracks from Celeritas tracks being handed back.
 *
 * This thread-local class receives the end-of-step state of the tracks that
 * were marked with \c SimTrackView::hand_back (and killed) during a step, and
 * creates a new \c G4Track for each one using \c
 * GeantTrackReconstruction::create . The reconstructed tracks are kept, in
 * track slot order, until they are retrieved with \c exchange_tracks .
 *
 * Host step data is copied immediately by the call operator. For device step
 * data, the call operator is invoked during the asynchronous launch of the
 * step: it enqueues the compaction of the selected tracks and the copy of
 * their number to pinned host memory, without synchronizing, and retains a
 * reference to the gathered step state. In both cases, after the producing
 * step is complete, and before another step can overwrite the step state, the
 * caller must call \c process_pending_steps , which reconstructs the tracks.
 * Hits from the same step must be processed first, since reconstructing the
 * offloaded track transfers its user information to the new \c G4Track . Only
 * one step can be pending.
 *
 * The kinematic state of each track is its post-step state: position,
 * direction, kinetic energy, global time, and weight. The vertex is set to the
 * same state, since Celeritas does not store the creation vertex of its
 * tracks. The touchable is reconstructed from the post-step volume path, so
 * the user stacking action sees a valid origin touchable. A track that is
 * outside the world after its step cannot be tracked by Geant4: it is
 * discarded, with a warning the first time.
 *
 * \warning This class \b must be created, used, and destroyed on a single
 * worker thread because of Geant4 thread-local allocators.
 */
class HandBackProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPTrackReconstruction = std::shared_ptr<GeantTrackReconstruction>;
    using VecTrack = std::vector<HandedBackTrack>;
    //!@}

  public:
    // Construct with track reconstruction
    explicit HandBackProcessor(SPTrackReconstruction track_reconstruction);

    ~HandBackProcessor();
    CELER_DELETE_COPY_MOVE(HandBackProcessor);

    // Copy CPU-generated handed-back tracks
    void operator()(StepStateHostRef const&);

    // Enqueue compaction of device-generated handed-back tracks
    void operator()(StepStateDeviceRef const&);

    // Reconstruct handed-back tracks after the step completes
    void process_pending_steps();

    //! Whether device-generated data is pending
    bool has_pending_steps() const noexcept
    {
        return pending_host_steps_ || static_cast<bool>(pending_device_steps_);
    }

    // Reconstruct tracks from a compacted step output (for testing)
    void operator()(DetectorStepOutput const& out);

    // Retrieve the reconstructed tracks, leaving none
    VecTrack exchange_tracks();

    //! Number of reconstructed tracks not yet retrieved
    size_type num_tracks() const { return tracks_.size(); }

    //! Number of tracks discarded because they were outside the world
    size_type num_outside() const { return num_outside_; }

    //! Access track reconstruction
    SPTrackReconstruction const& track_reconstruction() const
    {
        return track_reconstruction_;
    }

  private:
    template<class T>
    using PinnedVec = std::vector<T, PinnedAllocator<T>>;

    //! Track identities and per-primary data
    SPTrackReconstruction track_reconstruction_;
    //! Touchable reconstruction from volume instances
    std::unique_ptr<TouchableUpdaterInterface> update_touchable_;
    //! Device step data awaiting transfer after step completion
    StepStateDeviceRef pending_device_steps_;
    //! Whether host step data was copied and awaits reconstruction
    bool pending_host_steps_{false};
    //! Number of compacted device tracks (pinned for asynchronous copy)
    PinnedVec<size_type> num_selected_;
    //! Temporary CPU step information
    DetectorStepOutput steps_;
    //! Reconstructed tracks
    VecTrack tracks_;
    //! Number of tracks discarded outside the world
    size_type num_outside_{0};

    // Reconstruct a single track
    HandedBackTrack reconstruct(DetectorStepOutput const& out, size_type i);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
