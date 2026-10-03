//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HandBackProcessor.cc
//---------------------------------------------------------------------------//
#include "HandBackProcessor.hh"

#include <utility>
#include <G4LogicalVolume.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayQuantity.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantGeoParams.hh"

#include "LevelTouchableUpdater.hh"
#include "../GeantTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with track reconstruction and touchable option.
 */
HandBackProcessor::HandBackProcessor(
    SPTrackReconstruction track_reconstruction, bool locate_touchable)
    : track_reconstruction_{std::move(track_reconstruction)}
    , num_selected_(1, 0)
{
    CELER_EXPECT(track_reconstruction_);

    if (locate_touchable)
    {
        auto ggeo = ::celeritas::global_geant_geo().lock();
        CELER_VALIDATE(ggeo,
                       << "Geant4 geometry is required to reconstruct "
                          "touchables of handed-back tracks");
        update_touchable_
            = std::make_unique<LevelTouchableUpdater>(std::move(ggeo));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Release tracks that were never retrieved.
 */
HandBackProcessor::~HandBackProcessor()
{
    if (!tracks_.empty())
    {
        CELER_LOG_LOCAL(warning) << "Discarding " << tracks_.size()
                                 << " tracks that were not handed back";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Reconstruct CPU-generated handed-back tracks.
 */
void HandBackProcessor::operator()(StepStateHostRef const& states)
{
    copy_steps(&steps_, states);
    (*this)(steps_);
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue compaction of device-generated handed-back tracks.
 *
 * This is called during the asynchronous launch of a step: it does not
 * synchronize the stream. The step state remains valid until the next step
 * starts.
 */
void HandBackProcessor::operator()(StepStateDeviceRef const& states)
{
    CELER_EXPECT(states);
    CELER_EXPECT(!pending_device_steps_);

    compact_steps_async(states, num_selected_.data());
    pending_device_steps_ = states;
}

//---------------------------------------------------------------------------//
/*!
 * Copy and reconstruct device-generated tracks after their step completes.
 *
 * The caller must establish device step completion before calling this
 * function, which guarantees that the number of selected tracks has been
 * copied to the host. If no track was handed back during the step, no further
 * data is copied.
 */
void HandBackProcessor::process_pending_steps()
{
    if (!pending_device_steps_)
    {
        return;
    }

    auto states = std::exchange(pending_device_steps_, {});
    size_type const num_selected = num_selected_.front();
    if (num_selected == 0)
    {
        return;
    }
    copy_compacted_steps(&steps_, states, num_selected);
    (*this)(steps_);
}

//---------------------------------------------------------------------------//
/*!
 * Reconstruct tracks from a compacted step output.
 *
 * In an application setting, this is always called with our local data \c
 * steps_ as an argument. For tests, we can call this function explicitly using
 * local test data.
 */
void HandBackProcessor::operator()(DetectorStepOutput const& out)
{
    if (!out)
    {
        return;
    }

    ScopedProfiling profile_this{"reconstruct-hand-back"};
    tracks_.reserve(tracks_.size() + out.size());
    for (auto i : range(out.size()))
    {
        tracks_.push_back(this->reconstruct(out, i));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the reconstructed tracks, leaving none.
 */
auto HandBackProcessor::exchange_tracks() -> VecTrack
{
    return std::exchange(tracks_, {});
}

//---------------------------------------------------------------------------//
/*!
 * Reconstruct a single track.
 */
HandedBackTrack HandBackProcessor::reconstruct(DetectorStepOutput const& out,
                                               size_type i)
{
    CELER_EXPECT(i < out.size());
    CELER_EXPECT(i < out.particle_id.size() && i < out.primary_id.size()
                 && i < out.parent_id.size());

    auto const& post = out.points[StepPoint::post];
    CELER_ASSERT(i < post.pos.size() && i < post.dir.size()
                 && i < post.energy.size() && i < post.time.size());

    HandedBackTrack result;
    result.celer_track_id = out.track_id[i];
    result.celer_parent_id = out.parent_id[i];
    result.origin = result.celer_parent_id ? TrackOrigin::secondary
                                           : TrackOrigin::offloaded;
    if (!out.hand_back_reason.empty())
    {
        result.reason = out.hand_back_reason[i];
    }

    result.track = track_reconstruction_->create(
        out.particle_id[i], out.primary_id[i], result.origin);
    CELER_ASSERT(result.track);

    // Set kinematic state at the end of the step
    {
        using GTV = GeantTrackViewMutable;
        GTV gtv{*result.track};
        gtv.pos(native_value_to<GTV::Length>(post.pos[i]));
        gtv.dir(static_array_cast<double>(post.dir[i]));
        gtv.energy(GTV::Energy{static_cast<double>(post.energy[i].value())});
        gtv.time(native_value_to<GTV::Time>(post.time[i]));
        if (!out.weight.empty())
        {
            gtv.weight(out.weight[i]);
        }
    }

    // Celeritas does not track the creation vertex
    G4Track& track = *result.track;
    track.SetVertexPosition(track.GetPosition());
    track.SetVertexMomentumDirection(track.GetMomentumDirection());
    track.SetVertexKineticEnergy(track.GetKineticEnergy());

    if (update_touchable_)
    {
        auto* touchable = new G4TouchableHistory;
        G4TouchableHandle handle{touchable};
        if ((*update_touchable_)(out, i, StepPoint::post, touchable))
        {
            track.SetTouchableHandle(handle);
            track.SetNextTouchableHandle(handle);
            if (auto* pv = touchable->GetVolume())
            {
                track.SetLogicalVolumeAtVertex(pv->GetLogicalVolume());
            }
        }
        else
        {
            CELER_LOG_LOCAL(warning)
                << "Failed to reconstruct touchable for handed-back track "
                << result.celer_track_id.unchecked_get() << ": Geant4 will "
                << "locate it";
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
