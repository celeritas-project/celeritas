//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.test.cu
//---------------------------------------------------------------------------//
#include "ObserverPtr.test.hh"

#include <thrust/copy.h>

#include "corecel/device_runtime_api.h"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void copy_test_kernel(ObserverPtr<int const> in,
                                 ObserverPtr<int> out,
                                 size_type size)
{
    auto thread_idx = KernelParamCalculator::thread_id().unchecked_get();
    if (thread_idx >= size)
        return;

    out.get()[thread_idx] = in.get()[thread_idx];
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
void copy_test(ObserverPtr<int const, MemSpace::device> in_data,
               ObserverPtr<int, MemSpace::device> out_data,
               size_type size)
{
    CELER_LAUNCH_KERNEL(
        copy_test, device().default_block_size(), size, in_data, out_data, size);

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
void copy_thrust_test(ObserverPtr<int const, MemSpace::device> in_data,
                      ObserverPtr<int, MemSpace::device> out_data,
                      size_type size)
{
    thrust::copy(device_pointer_cast(in_data),
                 device_pointer_cast(in_data) + size,
                 device_pointer_cast(out_data));

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
