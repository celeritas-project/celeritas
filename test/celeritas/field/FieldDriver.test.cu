//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriver.test.cu
//---------------------------------------------------------------------------//
#include "FieldDriver.test.hh"

#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/Constants.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MagFieldTraits.hh"
#include "celeritas/field/UniformMagField.hh"

#include "FieldTestParams.hh"

using celeritas::MagFieldTraits;
using thrust::raw_pointer_cast;
using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void driver_test_kernel(const FieldDriverOptions data,
                                   FieldTestParams          test_params,
                                   double*                  pos_x,
                                   double*                  pos_z,
                                   double*                  mom_y,
                                   double*                  mom_z,
                                   double*                  error)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= test_params.nstates)
        return;

    // Construct the driver
    UniformMagField field({0, 0, test_params.field_value});
    using RKTraits
        = celeritas::MagFieldTraits<UniformMagField, DormandPrinceStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(data, rk4);

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
            auto end = driver.advance(hstep, y);
            total_step_length += end.step;
            y = end.state;
        }
    }

    // output for validation
    pos_x[tid.get()] = y.pos[0];
    pos_z[tid.get()] = y.pos[2];
    mom_y[tid.get()] = y.mom[1];
    mom_z[tid.get()] = y.mom[2];
    error[tid.get()] = total_step_length;
}

__global__ void accurate_advance_kernel(const FieldDriverOptions data,
                                        FieldTestParams          test_params,
                                        double*                  pos_x,
                                        double*                  pos_z,
                                        double*                  mom_y,
                                        double*                  mom_z,
                                        double*                  length)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= test_params.nstates)
        return;

    // Construct the driver
    UniformMagField field({0, 0, test_params.field_value});
    using RKTraits
        = celeritas::MagFieldTraits<UniformMagField, DormandPrinceStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(data, rk4);

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
            auto end = driver.accurate_advance(hstep, y_accurate, 0.001);
            total_curved_length += end.step;
            y_accurate = end.state;
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
driver_test(const FieldDriverOptions& fd_data, FieldTestParams test_params)
{
    // Input/Output data for kernel

    // Output data for kernel
    thrust::device_vector<double> pos_x(test_params.nstates, 0.0);
    thrust::device_vector<double> pos_z(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_y(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_z(test_params.nstates, 0.0);
    thrust::device_vector<double> error(test_params.nstates, 0.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(driver_test,
                        celeritas::device().default_block_size(),
                        test_params.nstates,
                        fd_data,
                        test_params,
                        raw_pointer_cast(pos_x.data()),
                        raw_pointer_cast(pos_z.data()),
                        raw_pointer_cast(mom_y.data()),
                        raw_pointer_cast(mom_z.data()),
                        raw_pointer_cast(error.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

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

OneGoodStepOutput accurate_advance_test(const FieldDriverOptions& fd_data,
                                        FieldTestParams           test_params)
{
    // Input/Output data for kernel

    // Output data for kernel
    thrust::device_vector<double> pos_x(test_params.nstates, 0.0);
    thrust::device_vector<double> pos_z(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_y(test_params.nstates, 0.0);
    thrust::device_vector<double> mom_z(test_params.nstates, 0.0);
    thrust::device_vector<double> length(test_params.nstates, 0.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(accurate_advance,
                        celeritas::device().default_block_size(),
                        test_params.nstates,
                        fd_data,
                        test_params,
                        raw_pointer_cast(pos_x.data()),
                        raw_pointer_cast(pos_z.data()),
                        raw_pointer_cast(mom_y.data()),
                        raw_pointer_cast(mom_z.data()),
                        raw_pointer_cast(length.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

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
