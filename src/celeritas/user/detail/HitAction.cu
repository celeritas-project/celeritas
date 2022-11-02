//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/HitAction.cu
//---------------------------------------------------------------------------//
#include "HitAction.cuh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void hit_action_kernel(CoreDeviceRef const             core,
                                  DeviceCRef<HitParamsData> const hit_params,
                                  DeviceRef<HitStateData> const   hit_state)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < data.states.size()))
        return;

    HitLauncher launch{core, hit_params, hit_state};
    launch(tid);
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void HitAction::execute(CoreDeviceRef const& core) const
{
    CELER_EXPECT(data);
    CELER_EXPECT(hit_state.size() == core.states.size());
    CELER_LAUNCH_KERNEL(hit_action,
                        celeritas::device().default_block_size(),
                        core.states.size(),
                        core,
                        hit_params,
                        hit_state);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
