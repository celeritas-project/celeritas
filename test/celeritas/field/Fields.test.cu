//---------------------------------*-CUDA-*----------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.cu
//---------------------------------------------------------------------------//
#include "Fields.test.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
namespace
{

using CartDeviceCRef = CartMapFieldParams::DeviceRef;
using RZDeviceCRef = RZMapFieldParams::DeviceRef;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void field_test_kernel(unsigned int const size,
                                  CartDeviceCRef field_map_data,
                                  inp::AxisGrid<double> x_grid,
                                  inp::AxisGrid<double> y_grid,
                                  inp::AxisGrid<double> z_grid,
                                  Array<size_type, 3> n_samples,
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
    Interpolator interp_x(
        {0, x_grid.min}, {static_cast<real_type>(nx_samples - 1), x_grid.max});
    Interpolator interp_y(
        {0, y_grid.min}, {static_cast<real_type>(ny_samples - 1), y_grid.max});
    Interpolator interp_z(
        {0, z_grid.min}, {static_cast<real_type>(nz_samples - 1), z_grid.max});

    for (size_type ix = 0; ix < nx_samples; ++ix)
    {
        real_type x = interp_x(ix);
        x = celeritas::min<real_type>(x, x_grid.max - 1);
        for (size_type iy = 0; iy < ny_samples; ++iy)
        {
            real_type y = interp_y(iy);
            y = celeritas::min<real_type>(y, y_grid.max - 1);
            for (size_type iz = 0; iz < nz_samples; ++iz)
            {
                real_type z = interp_z(iz);
                z = celeritas::min<real_type>(z, z_grid.max - 1);

                Real3 field = calc_field({x, y, z});
                field_values[index++] = field[0];
                field_values[index++] = field[1];
                field_values[index++] = field[2];
            }
        }
    }
}

__global__ void rzfield_test_kernel(unsigned int const num_points,
                                    RZDeviceCRef field_map_data,
                                    Real3 const* points,
                                    real_type* field_values)
{
    auto tid = TrackSlotId{KernelParamCalculator::thread_id().unchecked_get()};
    if (tid.get() >= num_points)
        return;

    RZMapField calc_field(field_map_data);
    Real3 field = calc_field(points[tid.get()]);
    field_values[tid.get() * 3 + 0] = field[0];
    field_values[tid.get() * 3 + 1] = field[1];
    field_values[tid.get() * 3 + 2] = field[2];
}

}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run CartMapField on device and return results
void field_test(inp::CartMapField& inp,
                Span<real_type>& field_values,
                Array<size_type, 3>& n_samples)
{
    CartMapFieldParams field_map{inp};

    DeviceVector<real_type> field_values_d(field_values.size());

    CartDeviceCRef device_cref = field_map.device_ref();

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
//! Run RZMapField on device and return results
void rzfield_test(RZMapFieldInput const& inp,
                  Span<Real3 const> points,
                  Span<real_type> field_values)
{
    CELER_EXPECT(field_values.size() == points.size() * 3);

    RZMapFieldParams field_map{inp};

    DeviceVector<Real3> points_d(points.size());
    points_d.copy_to_device(points);

    DeviceVector<real_type> field_values_d(field_values.size());

    CELER_LAUNCH_KERNEL(rzfield_test,
                        points.size(),
                        0,
                        static_cast<unsigned int>(points.size()),
                        field_map.device_ref(),
                        points_d.data(),
                        field_values_d.data());
    CELER_DEVICE_API_CALL(DeviceSynchronize());

    field_values_d.copy_to_host(field_values);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
