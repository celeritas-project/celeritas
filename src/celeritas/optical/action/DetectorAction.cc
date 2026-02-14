//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include "corecel/data/CollectionAlgorithms.hh"
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

    auto all_hits
        = state.ref()
              .detectors.detector_hits[AllItems<DetectorHit, MemSpace::host>{}];

    VecHit temp_hits(all_hits.size());
    // Copy all valid hits, erasing remaining part of the vector
    temp_hits.erase(
        std::copy_if(
            all_hits.begin(), all_hits.end(), temp_hits.begin(), Identity{}),
        temp_hits.end());
    this->callback_hits(temp_hits);
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
    -> VecHit
{
    auto const& native_hits = state.ref().detectors.detector_hits;
    VecHit temp_hits(native_hits.size());

    // Ensure the kernel copied into the device buffer before copying out
    celeritas::device().stream(state.stream_id()).sync();

    // Copy all track hits to host from device
    copy_to_host(native_hits, make_span(temp_hits), state.stream_id());

    // Ensure copy is complete
    celeritas::device().stream(state.stream_id()).sync();

    // Erase all hits with invalid detector ID
    temp_hits.erase(
        std::remove_if(temp_hits.begin(), temp_hits.end(), LogicalNot{}),
        temp_hits.end());

    return temp_hits;
}

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * Copied hits might be invalid, and are removed before sending into the
 * callback function. The callback is only executed when a non-zero number of
 * valid hits occurs.
 */
void DetectorAction::callback_hits(VecHit const& hits) const
{
    // Send hits to the callback function, if there are any
    if (!hits.empty())
    {
        callback_(make_span(hits));
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
