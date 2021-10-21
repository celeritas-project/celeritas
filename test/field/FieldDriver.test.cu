//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldDriver.test.cu
//---------------------------------------------------------------------------//
#include "FieldDriver.test.hh"
#include "FieldTestParams.hh"
#include "detail/MagTestTraits.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include <thrust/device_vector.h>

#include "field/FieldDriver.hh"
#include "field/FieldParamsData.hh"
#include "field/RungeKuttaStepper.hh"
#include "field/UniformMagField.hh"
#include "field/MagFieldEquation.hh"

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

__global__ void driver_test_kernel(const FieldParamsData pointers,
                                   FieldTestParams       test_params,
                                   double*               pos_x,
                                   double*               pos_z,
                                   double*               mom_y,
                                   double*               mom_z,
                                   double*               error)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= test_params.nstates)
        return;

    // Construct the driver
    UniformMagField field({0, 0, test_params.field_value});
    using RKTraits = detail::MagTestTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(pointers, rk4);

    // Test parameters and the sub-step size
    real_type hstep = 2 * constants::pi * test_params.radius
                      / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, tid.get() * 1.0e-6};
    y.mom = {0, test_params.momentum_y, test_params.momentum_z};

    // The rhs of the equation and a temporary array

    real_type total_step_length{0};

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_params.revolutions))
    {
        // Travel hstep for num_steps times in the field
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test_params.nsteps))
        {
            total_step_length += driver(hstep, &y);
        }
        // Check the total length
    }

    // output for validation
    pos_x[tid.get()] = y.pos[0];
    pos_z[tid.get()] = y.pos[2];
    mom_y[tid.get()] = y.mom[1];
    mom_z[tid.get()] = y.mom[2];
    error[tid.get()] = total_step_length;
}

__global__ void accurate_advance_kernel(const FieldParamsData pointers,
                                        FieldTestParams       test_params,
                                        double*               pos_x,
                                        double*               pos_z,
                                        double*               mom_y,
                                        double*               mom_z,
                                        double*               length)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= test_params.nstates)
        return;

    // Construct the driver
    UniformMagField field({0, 0, test_params.field_value});
    using RKTraits = detail::MagTestTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(pointers, rk4);

    // Test parameters and the sub-step size
    real_type circumference = 2 * constants::pi * test_params.radius;
    real_type hstep         = circumference / test_params.nsteps;

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {test_params.radius, 0, tid.get() * 1.0e-6};
    y.mom = {0, test_params.momentum_y, test_params.momentum_z};

    // The rhs of the equation and a temporary array
    OdeState y_accurate;

    real_type total_curved_length{0};

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_params.revolutions))
    {
        // test quick_advance
        y_accurate = y;

        // Travel hstep for num_steps times in the field
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test_params.nsteps))
        {
            total_curved_length
                += driver.accurate_advance(hstep, &y_accurate, 0.001);
        }
    }

    // output for validation
    pos_x[tid.get()]  = y_accurate.pos[0];
    pos_z[tid.get()]  = y_accurate.pos[2];
    mom_y[tid.get()]  = y_accurate.mom[1];
    mom_z[tid.get()]  = y_accurate.mom[2];
    length[tid.get()] = total_curved_length;
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
FITestOutput
driver_test(const FieldParamsData& fd_pointers, FieldTestParams test_params)
{
    // Input/Output data for kernel

    // Output data for kernel
    thrust::device_vector<double> pos_x(test_params.nstates, 0.0);
    thrust::device_vector<double> pos_z(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_y(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_z(test_params.nstates, 0.0);
    thrust::device_vector<double> error(test_params.nstates, 0.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(driver_test_kernel,
                                                        "driver_test");
    auto params = calc_launch_params(test_params.nstates);

    driver_test_kernel<<<params.grid_size, params.block_size>>>(
        fd_pointers,
        test_params,
        raw_pointer_cast(pos_x.data()),
        raw_pointer_cast(pos_z.data()),
        raw_pointer_cast(mom_y.data()),
        raw_pointer_cast(mom_z.data()),
        raw_pointer_cast(error.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    FITestOutput result;

    result.pos_x.resize(pos_x.size());
    thrust::copy(pos_x.begin(), pos_x.end(), result.pos_x.begin());

    result.pos_z.resize(pos_z.size());
    thrust::copy(pos_z.begin(), pos_z.end(), result.pos_z.begin());

    result.mom_y.resize(mom_y.size());
    thrust::copy(mom_y.begin(), mom_y.end(), result.mom_y.begin());

    result.mom_z.resize(mom_z.size());
    thrust::copy(mom_z.begin(), mom_z.end(), result.mom_z.begin());

    result.error.resize(error.size());
    thrust::copy(error.begin(), error.end(), result.error.begin());
    return result;
}

OneGoodStepOutput accurate_advance_test(const FieldParamsData& fd_pointers,
                                        FieldTestParams        test_params)
{
    // Input/Output data for kernel

    // Output data for kernel
    thrust::device_vector<double> pos_x(test_params.nstates, 0.0);
    thrust::device_vector<double> pos_z(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_y(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_z(test_params.nstates, 0.0);
    thrust::device_vector<double> length(test_params.nstates, 0.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(
        accurate_advance_kernel, "accurate_advance_test");
    auto params = calc_launch_params(test_params.nstates);

    accurate_advance_kernel<<<params.grid_size, params.block_size>>>(
        fd_pointers,
        test_params,
        raw_pointer_cast(pos_x.data()),
        raw_pointer_cast(pos_z.data()),
        raw_pointer_cast(mom_y.data()),
        raw_pointer_cast(mom_z.data()),
        raw_pointer_cast(length.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    OneGoodStepOutput result;

    result.pos_x.resize(pos_x.size());
    thrust::copy(pos_x.begin(), pos_x.end(), result.pos_x.begin());

    result.pos_z.resize(pos_z.size());
    thrust::copy(pos_z.begin(), pos_z.end(), result.pos_z.begin());

    result.mom_y.resize(mom_y.size());
    thrust::copy(mom_y.begin(), mom_y.end(), result.mom_y.begin());

    result.mom_z.resize(mom_z.size());
    thrust::copy(mom_z.begin(), mom_z.end(), result.mom_z.begin());

    result.length.resize(length.size());
    thrust::copy(length.begin(), length.end(), result.length.begin());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
