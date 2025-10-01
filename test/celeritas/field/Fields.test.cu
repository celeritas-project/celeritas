//---------------------------------*-CUDA-*----------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.cu
//---------------------------------------------------------------------------//
#include "Fields.test.hh"

#include <cstdio>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
namespace
{

using DeviceCRef = CartMapFieldParams::DeviceRef;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void field_test_kernel(unsigned int const size,
                                  DeviceCRef field_map_data,
                                  AxisGrid<real_type> x_grid,
                                  AxisGrid<real_type> y_grid,
                                  AxisGrid<real_type> z_grid,
                                  Real3 n_samples,
                                  real_type* field_values)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() >= size)
        return;

    CartMapField calc_field(field_map_data);

    size_type nx_samples = static_cast<size_type>(n_samples[0]);
    size_type ny_samples = static_cast<size_type>(n_samples[1]);
    size_type nz_samples = static_cast<size_type>(n_samples[2]);

    size_type index = 0;

    for (size_type ix = 0; ix < nx_samples; ++ix)
    {
        real_type x = x_grid.min
                      + ix * (x_grid.max - x_grid.min) / (nx_samples - 1);
        x = celeritas::min(x, x_grid.max - 1);
        for (size_type iy = 0; iy < ny_samples; ++iy)
        {
            real_type y = y_grid.min
                          + iy * (y_grid.max - y_grid.min) / (ny_samples - 1);
            y = celeritas::min(y, y_grid.max - 1);
            for (size_type iz = 0; iz < nz_samples; ++iz)
            {
                real_type z = z_grid.min
                              + iz * (z_grid.max - z_grid.min)
                                    / (nz_samples - 1);
                z = celeritas::min(z, z_grid.max - 1);

                Real3 field = calc_field({x, y, z});
                field_values[index++] = field[0];
                field_values[index++] = field[1];
                field_values[index++] = field[2];
            }
        }
    }
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void field_test(CartMapFieldInput& inp,
                Span<real_type>& field_values,
                Real3& n_samples)
{
    CartMapFieldParams field_map{inp};

    DeviceVector<real_type> field_values_d(field_values.size());

    DeviceCRef device_cref = field_map.device_ref();

    CELER_LAUNCH_KERNEL(field_test,
                        1,
                        0,
                        1,
                        device_cref,
                        inp.x,
                        inp.y,
                        inp.z,
                        n_samples,
                        field_values_d.data());
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    field_values_d.copy_to_host(field_values);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
