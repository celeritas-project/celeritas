//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserParamField.test.cu
//---------------------------------------------------------------------------//
#include "UserField.test.hh"
#include "detail/CMSParameterizedField.hh"

#include <thrust/device_vector.h>

#include "base/KernelParamCalculator.device.hh"
#include "base/device_runtime_api.h"
#include "comm/Device.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void parameterized_field_test_kernel(UserFieldTestParams param,
                                                real_type*          value_x,
                                                real_type*          value_y,
                                                real_type*          value_z)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= param.nsamples)
        return;

    detail::CMSParameterizedField field;
    Real3                         pos{tid.get() * param.delta_r,
              tid.get() * param.delta_r,
              tid.get() * param.delta_z};
    Real3                         value = field(pos);

    // Output for verification
    value_x[tid.get()] = value[0];
    value_y[tid.get()] = value[1];
    value_z[tid.get()] = value[2];
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
UserFieldTestOutput parameterized_field_test(UserFieldTestParams test_param)
{
    // Output data for kernel
    thrust::device_vector<real_type> value_x(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_y(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_z(test_param.nsamples, 0.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(parameterized_field_test,
                        celeritas::device().default_block_size(),
                        test_param.nsamples,
                        test_param,
                        raw_pointer_cast(value_x.data()),
                        raw_pointer_cast(value_y.data()),
                        raw_pointer_cast(value_z.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy result back to CPU
    UserFieldTestOutput result;

    result.value_x.resize(value_x.size());
    thrust::copy(value_x.begin(), value_x.end(), result.value_x.begin());

    result.value_y.resize(value_y.size());
    thrust::copy(value_y.begin(), value_y.end(), result.value_y.begin());

    result.value_z.resize(value_z.size());
    thrust::copy(value_z.begin(), value_z.end(), result.value_z.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
