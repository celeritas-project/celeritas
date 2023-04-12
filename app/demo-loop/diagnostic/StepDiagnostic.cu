//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/diagnostic/StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "corecel/sys/KernelParamCalculator.device.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Count the steps per track for each particle type.
 */
__global__ void count_steps_kernel(DeviceRef<CoreParamsData> const params,
                                   DeviceRef<CoreStateData> const states,
                                   StepDiagnosticDataRef<MemSpace::device> data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    StepLauncher<MemSpace::device> launch(params, states, data);
    launch(TrackSlotId{tid.unchecked_get()});
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Launch kernel to tally the steps per track.
 */
void count_steps(DeviceRef<CoreParamsData> const& params,
                 DeviceRef<CoreStateData> const& states,
                 StepDiagnosticDataRef<MemSpace::device> data)
{
    CELER_LAUNCH_KERNEL(count_steps,
                        celeritas::device().default_block_size(),
                        states.size(),
                        params,
                        states,
                        data);
}
//---------------------------------------------------------------------------//
}  // namespace demo_loop
