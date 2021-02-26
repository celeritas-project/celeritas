//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.cu
//---------------------------------------------------------------------------//
#include "FieldPropagator.test.hh"
#include "FieldTestParams.hh"

#include "field/base/FieldTrackView.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"

#include "field/FieldParamsPointers.hh"
#include "field/MagField.hh"
#include "field/FieldEquation.hh"
#include "field/RungeKutta.hh"
#include "field/FieldIntegrator.hh"
#include "field/FieldPropagator.hh"

#include <thrust/device_vector.h>

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void fp_test_kernel(const int                  size,
                               const GeoParamsPointers    shared,
                               const GeoStatePointers     state,
                               const GeoStateInitializer* start,
                               ParticleParamsPointers     particle_params,
                               ParticleStatePointers      particle_states,
                               FieldParamsPointers        field_params,
                               FieldTestParams            test_params,
                               const ParticleTrackState*  init_track,
                               double*                    pos,
                               double*                    mom,
                               double*                    step)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize geo
    GeoTrackView geo(shared, state, tid);
    geo = start[tid.get()];

    // Initialize particle
    ParticleTrackView p(particle_params, particle_states, tid);
    p = init_track[tid.get()];

    if (!geo.is_outside())
    {
        geo.find_next_step();
    }

    // Construct FieldTrackView
    FieldTrackView field_view(geo, p);

    // Construct the RK stepper adnd propagator in a field
    const Real3 bfield{0, 0, test_params.field_value}; // a uniform B-field

    MagField        magfield(bfield);
    FieldEquation   equation(magfield);
    RungeKutta      rk4(equation);
    FieldIntegrator integrator(field_params, rk4);

    FieldPropagator propagator(field_params, integrator);

    // Tests with input parameters of a electron in a uniform magnetic field
    double hstep = (2.0 * constants::pi * test_params.radius)
                   / test_params.nsteps;

    real_type curved_length = 0;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_params.revolutions))
    {
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test_params.nsteps))
        {
            field_view.h() = hstep;
            curved_length += propagator(field_view);
        }
    }

    // output
    step[tid.get()] = curved_length;
    pos[tid.get()]  = field_view.y()[0];
    mom[tid.get()]  = field_view.y()[4];
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
FPTestOutput fp_test(FPTestInput input)
{
    CELER_ASSERT(input.init_geo.size() == input.init_track.size());
    CELER_ASSERT(input.geo_params);
    CELER_ASSERT(input.geo_states);
    CELER_ASSERT(input.init_geo.size() == input.geo_states.size);

    // Temporary device data for kernel
    thrust::device_vector<GeoStateInitializer> in_geo   = input.init_geo;
    thrust::device_vector<ParticleTrackState>  in_track = input.init_track;

    // Output data for kernel
    thrust::device_vector<double> step(input.init_geo.size(), -1.0);
    thrust::device_vector<double> pos(input.init_geo.size(), -1.0);
    thrust::device_vector<double> mom(input.init_geo.size(), -1.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(fp_test_kernel,
                                                        "fp_test");
    auto params = calc_launch_params(in_geo.size());

    fp_test_kernel<<<params.grid_size, params.block_size>>>(
        in_geo.size(),
        input.geo_params,
        input.geo_states,
        raw_pointer_cast(in_geo.data()),
        input.particle_params,
        input.particle_states,
        input.field_params,
        input.test_params,
        raw_pointer_cast(in_track.data()),
        raw_pointer_cast(pos.data()),
        raw_pointer_cast(mom.data()),
        raw_pointer_cast(step.data()));

    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy result back to CPU
    FPTestOutput result;

    result.step.resize(step.size());
    thrust::copy(step.begin(), step.end(), result.step.begin());

    result.pos.resize(pos.size());
    thrust::copy(pos.begin(), pos.end(), result.pos.begin());

    result.mom.resize(mom.size());
    thrust::copy(mom.begin(), mom.end(), result.mom.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
