//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldIntegrator.test.cu
//---------------------------------------------------------------------------//
#include "FieldIntegrator.test.hh"
#include "FieldTestParams.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include <thrust/device_vector.h>

#include "field/FieldIntegrator.hh"
#include "field/FieldParamsPointers.hh"
#include "field/RungeKutta.hh"
#include "field/MagField.hh"
#include "field/FieldEquation.hh"

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

__global__ void integrator_test_kernel(const FieldParamsPointers pointers,
                                       FieldTestParams           test_params,
                                       double*                   pos,
                                       double*                   mom,
                                       double*                   err)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= test_params.nstates)
        return;

    // Construct the integrator
    MagField        field({0, 0, test_params.field_value});
    FieldEquation   equation(field);
    RungeKutta      rk4(equation);
    FieldIntegrator integrator(pointers, rk4);

    // Initial state and the epected state after revolutions
    OdeArray y;
    y[0] = test_params.radius;
    y[4] = test_params.momentum;

    // The rhs of the equation and a temporary array
    OdeArray dydx;

    // Test parameters and the sub-step size
    real_type circumference = 2.0 * constants::pi * test_params.radius;

    test_params.nsteps = 100;
    double hstep       = circumference / test_params.nsteps;

    real_type curve_length = 0;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_params.revolutions))
    {
        // Travel hstep for num_steps times in the field
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test_params.nsteps))
        {
            dydx = equation(y);
            integrator.accurate_advance(hstep, y, curve_length, 0.001);
        }

        // XXX TODO: Add a test for quick_advance
    }

    // output for validation
    pos[tid.get()] = y[0];
    mom[tid.get()] = y[4];
    err[tid.get()] = curve_length;
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
FDTestOutput integrator_test(const FieldParamsPointers& fd_pointers,
                             FieldTestParams            test_params)
{
    // Input parameters for device

    // Output data for kernel
    thrust::device_vector<double> pos(test_params.nstates, 0.0);
    thrust::device_vector<double> mom(test_params.nstates, 0.0);
    thrust::device_vector<double> err(test_params.nstates, 0.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(integrator_test_kernel,
                                                        "integrator_test");
    auto params = calc_launch_params(test_params.nstates);

    integrator_test_kernel<<<params.grid_size, params.block_size>>>(
        fd_pointers,
        test_params,
        raw_pointer_cast(pos.data()),
        raw_pointer_cast(mom.data()),
        raw_pointer_cast(err.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    FDTestOutput result;

    result.pos.resize(pos.size());
    thrust::copy(pos.begin(), pos.end(), result.pos.begin());

    result.mom.resize(mom.size());
    thrust::copy(mom.begin(), mom.end(), result.mom.begin());

    result.err.resize(err.size());
    thrust::copy(err.begin(), err.end(), result.err.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
