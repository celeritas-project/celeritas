//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RungeKutta.test.cu
//---------------------------------------------------------------------------//
#include "RungeKutta.test.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include <thrust/device_vector.h>

#include "field/MagField.hh"
#include "field/FieldEquation.hh"
#include "field/RungeKutta.hh"

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

__global__ void rk4_test_kernel(FieldTestParams param,
                                real_type*      pos_x,
                                real_type*      mom_y,
                                real_type*      error)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= param.nstates)
        return;

    // Construct the Runge-Kutta stepper
    MagField      field({0, 0, param.field_value});
    FieldEquation equation(field);
    RungeKutta    rk4(equation);

    // Initial state and the epected state after revolutions
    OdeArray y;
    y[0] = param.radius;
    y[2] = tid.get() * 1.0e-6; //!< XXX use random position here
    y[4] = param.momentum;

    // The rhs of the equation and a temporary array
    OdeArray dydx;
    OdeArray yout;

    // Test parameters and the sub-step size
    real_type hstep       = 2.0 * constants::pi * param.radius / param.nsteps;
    real_type total_error = 0;

    for (int nr = 0; nr < param.revolutions; ++nr)
    {
        // Travel hstep for nsteps times in the field
        for (CELER_MAYBE_UNUSED int i : celeritas::range(param.nsteps))
        {
            dydx            = equation(y);
            real_type error = rk4.stepper(hstep, y, dydx, yout);
            y               = yout;
            total_error += error;
        }
    }

    // Output for verification
    pos_x[tid.get()] = y[0];
    mom_y[tid.get()] = y[4];
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
    thrust::device_vector<real_type> mom_y(test_param.nstates, 0.0);
    thrust::device_vector<real_type> error(test_param.nstates, 0.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(rk4_test_kernel,
                                                        "rk4_test");
    auto params = calc_launch_params(test_param.nstates);

    rk4_test_kernel<<<params.grid_size, params.block_size>>>(
        test_param,
        raw_pointer_cast(pos_x.data()),
        raw_pointer_cast(mom_y.data()),
        raw_pointer_cast(error.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    RK4TestOutput result;

    result.pos_x.resize(pos_x.size());
    thrust::copy(pos_x.begin(), pos_x.end(), result.pos_x.begin());

    result.mom_y.resize(mom_y.size());
    thrust::copy(mom_y.begin(), mom_y.end(), result.mom_y.begin());

    result.error.resize(error.size());
    thrust::copy(error.begin(), error.end(), result.error.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
