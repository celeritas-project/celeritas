//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserMapField.test.cu
//---------------------------------------------------------------------------//
#include <thrust/device_vector.h>

#include "base/device_runtime_api.h"
#include "base/Constants.hh"
#include "base/KernelParamCalculator.device.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "comm/Device.hh"

#include "UserField.test.hh"
#include "detail/CMSMapField.hh"
#include "detail/FieldMapData.hh"
#include "detail/MagFieldMap.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void fieldmap_test_kernel(UserFieldTestParams       param,
                                     detail::FieldMapDeviceRef field_data,
                                     real_type*                value_x,
                                     real_type*                value_y,
                                     real_type*                value_z)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= param.nsamples)
        return;

    detail::CMSMapField field(field_data);
    //    Real3 pos{tid.get()*1.5-4, tid.get()*1.5-4, tid.get()*2.5-4};
    Real3 pos{tid.get() * param.delta_r,
              tid.get() * param.delta_r,
              tid.get() * param.delta_z};

    Real3 value = field(pos);

    // Output for verification
    value_x[tid.get()] = value[0];
    value_y[tid.get()] = value[1];
    value_z[tid.get()] = value[2];
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
UserFieldTestOutput fieldmap_test(UserFieldTestParams       test_param,
                                  detail::FieldMapDeviceRef field_data)
{
    // Output data for kernel
    thrust::device_vector<real_type> value_x(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_y(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_z(test_param.nsamples, 0.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(fieldmap_test,
                        celeritas::device().default_block_size(),
                        test_param.nsamples,
                        test_param,
                        field_data,
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
