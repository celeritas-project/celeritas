//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticImpl.cu
//---------------------------------------------------------------------------//
#include "ActionDiagnosticImpl.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

#include "ActionDiagnosticLauncher.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type on device.
 */
__global__ void tally_action_kernel(DeviceCRef<CoreParamsData> const params,
                                    DeviceRef<CoreStateData> const state,
                                    DeviceRef<ActionDiagnosticStateData> data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < state.size()))
        return;

    ActionDiagnosticLauncher launch{params, state, data};
    launch(tid);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type on device.
 */
void tally_action(DeviceCRef<CoreParamsData> const& params,
                  DeviceRef<CoreStateData> const& state,
                  DeviceRef<ActionDiagnosticStateData>& data)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL(tally_action,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params,
                        state,
                        data);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
