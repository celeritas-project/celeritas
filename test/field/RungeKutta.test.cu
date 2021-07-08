//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.cu
//---------------------------------------------------------------------------//
#include "RungeKutta.test.hh"
#include "detail/MagTestTraits.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include <thrust/device_vector.h>

#include "field/UniformMagField.hh"
#include "field/MagFieldEquation.hh"
#include "field/RungeKuttaStepper.hh"
#include "field/FieldInterface.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "base/Constants.hh"
#include "base/Units.hh"
#include "physics/base/Units.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void rk4_test_kernel(FieldTestParams param,
                                real_type*      pos_x,
                                real_type*      pos_z,
                                real_type*      mom_y,
                                real_type*      mom_z,
                                real_type*      error)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= param.nstates)
        return;

    // Construct the Runge-Kutta stepper
    UniformMagField field({0, 0, param.field_value});
    using RKTraits = detail::MagTestTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);

    // Initial state and the epected state after revolutions
    OdeState y;
    y.pos = {param.radius, 0.0, tid.get() * 1.0e-6};
    y.mom = {0.0, param.momentum_y, param.momentum_z};

    // Test parameters and the sub-step size
    real_type hstep       = 2.0 * constants::pi * param.radius / param.nsteps;
    real_type total_error = 0;

    for (auto nr : range(param.revolutions))
    {
        // Travel hstep for nsteps times in the field
        for (CELER_MAYBE_UNUSED int i : celeritas::range(param.nsteps))
        {
            StepperResult result = rk4(hstep, y);
            y                    = result.end_state;
            total_error += truncation_error(hstep, 0.001, y, result.err_state);
        }
    }

    // Output for verification
    pos_x[tid.get()] = y.pos[0];
    pos_z[tid.get()] = y.pos[2];
    mom_y[tid.get()] = y.mom[1];
    mom_z[tid.get()] = y.mom[2];
    error[tid.get()] = total_error;
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
RK4TestOutput rk4_test(FieldTestParams test_param)
{
    // Output data for kernel
    thrust::device_vector<real_type> pos_x(test_param.nstates, 0.0);
    thrust::device_vector<real_type> pos_z(test_param.nstates, 0.0);
    thrust::device_vector<real_type> mom_y(test_param.nstates, 0.0);
    thrust::device_vector<real_type> mom_z(test_param.nstates, 0.0);
    thrust::device_vector<real_type> error(test_param.nstates, 0.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(rk4_test_kernel,
                                                        "rk4_test");
    auto params = calc_launch_params(test_param.nstates);

    rk4_test_kernel<<<params.grid_size, params.block_size>>>(
        test_param,
        raw_pointer_cast(pos_x.data()),
        raw_pointer_cast(pos_z.data()),
        raw_pointer_cast(mom_y.data()),
        raw_pointer_cast(mom_z.data()),
        raw_pointer_cast(error.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    RK4TestOutput result;

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

//---------------------------------------------------------------------------//
} // namespace celeritas_test
