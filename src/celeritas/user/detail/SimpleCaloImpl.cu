//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SimpleCaloImpl.cu
//---------------------------------------------------------------------------//
#include "SimpleCaloImpl.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"

#include "SimpleCaloExecutor.hh"  // IWYU pragma: associated

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
 * Accumulate energy deposition on device.
 */
__global__ void simple_calo_accum_kernel(DeviceRef<StepStateData> const step,
                                         DeviceRef<SimpleCaloStateData> calo)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < step.size()))
        return;

    SimpleCaloExecutor execute{step, calo};
    execute(tid);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition on device.
 */
void simple_calo_accum(DeviceRef<StepStateData> const& step,
                       DeviceRef<SimpleCaloStateData>& calo)
{
    CELER_EXPECT(step && calo);
    CELER_LAUNCH_KERNEL(simple_calo_accum,
                        step.size(),
                        celeritas::device().stream(step.stream_id).get(),
                        step,
                        calo);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
