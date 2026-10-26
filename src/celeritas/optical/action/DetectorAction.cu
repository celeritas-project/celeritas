//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DetectorAction.cu
//---------------------------------------------------------------------------//
#include "DetectorAction.hh"

#include "corecel/data/AsyncCompactedCopy.device.t.hh"

#include "ActionLauncher.device.hh"
#include "TrackSlotExecutor.hh"

#include "detail/DetectorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the detector action on device.
 */
void DetectorAction::step(CoreParams const& params,
                          CoreStateDevice& state) const
{
    TrackSlotExecutor execute{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::DetectorExecutor{state.ref().detectors}};

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);

    // Copy the hits of the previous step and compact the hits of this one
    auto& hits = this->copy_hits_async(state);
    auto const& device_hits = state.ref().detectors.detector_hits;
    hits.compact_async(device_hits.data(), device_hits.size());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
