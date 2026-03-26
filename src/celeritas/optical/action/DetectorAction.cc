//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/DetectorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
DetectorAction::DetectorAction(ActionId aid, CallbackFunc const& callback)
    : StaticConcreteAction(
          aid, "optical-detector", "Score optical detector hits")
    , callback_(callback)
{
    CELER_EXPECT(callback);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the detector action on host.
 *
 * \todo avoid reallocating the temporary storage at every step, or as an
 * optimization just call contiguous chunks of hits.
 */
void DetectorAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::DetectorExecutor{state.ref().detectors}};
    launch_action(state, execute);

    auto const& det_state = state.ref().detectors;
    auto all_hits
        = det_state.detector_hits[AllItems<DetectorHit, MemSpace::host>{}];

    HitsOutput out;
    out.num_volume_levels = det_state.num_volume_levels;

    // Reserve to avoid repeated allocation
    out.hits.reserve(all_hits.size());
    if (out.num_volume_levels > 0)
    {
        out.volume_instance_ids.reserve(all_hits.size()
                                        * out.num_volume_levels);
    }

    auto all_vol_ids
        = det_state
              .volume_instance_ids[AllItems<VolumeInstanceId, MemSpace::host>{}];

    for (auto i : range(all_hits.size()))
    {
        if (all_hits[i])
        {
            out.hits.push_back(all_hits[i]);
            if (out.num_volume_levels > 0)
            {
                auto src = all_vol_ids.subspan(i * out.num_volume_levels,
                                               out.num_volume_levels);
                out.volume_instance_ids.insert(
                    out.volume_instance_ids.end(), src.begin(), src.end());
            }
        }
    }

    this->callback_hits(std::move(out));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void DetectorAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * \todo Replace this with asynchronous calls into pinned memory in aux
 * state, followed by an asynchronous callback.
 */
auto DetectorAction::load_hits_sync(CoreStateDevice const& state) const
    -> HitsOutput
{
    auto const& det_state = state.ref().detectors;
    auto const& native_hits = det_state.detector_hits;
    std::vector<DetectorHit> temp_hits(native_hits.size());

    // Copy all track hits to host from device (async, same stream as kernel)
    copy_to_host(native_hits, make_span(temp_hits), state.stream_id());

    // Copy volume hierarchy if allocated
    std::vector<VolumeInstanceId> temp_vol_ids;
    if (det_state.num_volume_levels > 0)
    {
        temp_vol_ids.resize(det_state.volume_instance_ids.size());
        copy_to_host(det_state.volume_instance_ids,
                     make_span(temp_vol_ids),
                     state.stream_id());
    }

    // Ensure copy is complete
    celeritas::device().stream(state.stream_id()).sync();

    // Filter valid hits and collect corresponding volume slices
    HitsOutput out;
    out.num_volume_levels = det_state.num_volume_levels;
    out.hits.reserve(temp_hits.size());
    if (out.num_volume_levels > 0)
    {
        out.volume_instance_ids.reserve(temp_hits.size()
                                        * out.num_volume_levels);
    }

    for (auto i : range(temp_hits.size()))
    {
        if (temp_hits[i])
        {
            out.hits.push_back(temp_hits[i]);
            if (out.num_volume_levels > 0)
            {
                auto beg
                    = temp_vol_ids.begin()
                      + static_cast<std::ptrdiff_t>(i * out.num_volume_levels);
                out.volume_instance_ids.insert(
                    out.volume_instance_ids.end(),
                    beg,
                    beg + static_cast<std::ptrdiff_t>(out.num_volume_levels));
            }
        }
    }

    return out;
}

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * Copied hits might be invalid, and are removed before sending into the
 * callback function. The callback is only executed when a non-zero number of
 * valid hits occurs.
 */
void DetectorAction::callback_hits(HitsOutput&& out) const
{
    // Send hits to the callback function, if there are any
    if (!out.hits.empty())
    {
        CELER_LOG_LOCAL(debug) << "Dispatching " << out.hits.size()
                               << " optical detector hits to callback";
        callback_(out);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
