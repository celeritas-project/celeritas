//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/generated/BoundaryAction.cu
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/BoundaryActionImpl.hh"

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void boundary_kernel(CoreDeviceRef const data
)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < data.states.size()))
        return;

    auto launch = make_track_launcher(data, detail::boundary_track);
    launch(TrackSlotId{tid.unchecked_get()});
}
}  // namespace

void BoundaryAction::execute(CoreDeviceRef const& data) const
{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(boundary,
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data);
}

}  // namespace generated
}  // namespace celeritas
