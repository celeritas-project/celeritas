//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorAction.cc
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "DetectorExecutor.hh"

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
 */
void DetectorAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              DetectorExecutor{state.ref().detectors}};
    launch_action(state, execute);

    this->process_hits(state);
}

#if !CELER_USE_DEVICE
void DetectorAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 */
void DetectorAction::process_hits(CoreStateHost& state) const
{
    this->process_hits_impl<MemSpace::host>(state);
}

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 */
void DetectorAction::process_hits(CoreStateDevice& state) const
{
    this->process_hits_impl<MemSpace::device>(state);
}

//---------------------------------------------------------------------------//
/*!
 * Process hits copied from the kernels and send them to the callback.
 *
 * Copied hits might be invalid, and are removed before sending into the
 * callback function. The callback is only execute when a non-zero amount of
 * valid hits occurs.
 */
template<MemSpace M>
void DetectorAction::process_hits_impl(CoreState<M>& state) const
{
    DetectorHitOutput hit_results;

    // Copy hits (possibly from device) into pinned vector
    copy_hits<M>(&hit_results, state.ref().detectors, state.stream_id());

    // Erase all hits with invalid detector ID
    hit_results.hits.erase(
        std::remove_if(hit_results.hits.begin(),
                       hit_results.hits.end(),
                       [](DetectorHit const& hit) {
                           return !static_cast<bool>(hit.detector);
                       }),
        hit_results.hits.end());

    // Send hits to the callback function, if there are any
    if (!hit_results.hits.empty())
    {
        callback_(make_span(hit_results.hits));
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
