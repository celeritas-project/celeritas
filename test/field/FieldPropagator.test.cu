//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.cu
//---------------------------------------------------------------------------//
#include "FieldTestParams.hh"
#include "FieldPropagator.test.hh"
#include "field/FieldTrackView.hh"
#include "field/FieldParamsPointers.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"

#include "field/MagField.hh"
#include "field/MagFieldEquation.hh"
#include "field/RungeKuttaStepper.hh"
#include "field/FieldDriver.hh"
#include "field/FieldPropagator.hh"

#include <thrust/device_vector.h>

using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void fp_test_kernel(const int                 size,
                               const GeoParamsCRefDevice shared,
                               const GeoStateRefDevice   state,
                               const VGGTestInit*        start,
                               ParticleParamsPointers    particle_params,
                               ParticleStatePointers     particle_states,
                               FieldParamsPointers       field_params,
                               FieldTestParams           test,
                               const ParticleTrackState* init_track,
                               double*                   pos,
                               double*                   mom,
                               double*                   step)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize GeoTrackView and ParticleTrackView
    GeoTrackView geo_view(shared, state, tid);
    geo_view = start[tid.get()];
    if (!geo_view.is_outside())
        geo_view.find_next_step();

    ParticleTrackView track_view(particle_params, particle_states, tid);
    track_view = init_track[tid.get()];

    // Construct FieldTrackView
    FieldTrackView field_view(geo_view, track_view);

    // Construct the RK stepper adnd propagator in a field
    MagField                            field({0, 0, test.field_value});
    MagFieldEquation                    equation(field, field_view.charge());
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);
    FieldPropagator                     propagator(field_params, driver);

    // Tests with input parameters of a electron in a uniform magnetic field
    double hstep = (2.0 * constants::pi * test.radius) / test.nsteps;

    real_type curved_length = 0;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test.revolutions))
    {
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test.nsteps))
        {
            field_view.step() = hstep;
            curved_length += propagator(&field_view);
        }
    }

    // output
    step[tid.get()] = curved_length;
    pos[tid.get()]  = field_view.state().pos[0];
    mom[tid.get()]  = field_view.state().mom[1];
}

__global__ void bc_test_kernel(const int                 size,
                               const GeoParamsCRefDevice shared,
                               const GeoStateRefDevice   state,
                               const VGGTestInit*        start,
                               ParticleParamsPointers    particle_params,
                               ParticleStatePointers     particle_states,
                               FieldParamsPointers       field_params,
                               FieldTestParams           test,
                               const ParticleTrackState* init_track,
                               double*                   pos,
                               double*                   mom,
                               double*                   step)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize GeoTrackView and ParticleTrackView
    GeoTrackView geo_view(shared, state, tid);
    geo_view = start[tid.get()];
    if (!geo_view.is_outside())
        geo_view.find_next_step();

    // Initialize particle
    ParticleTrackView track_view(particle_params, particle_states, tid);
    track_view = init_track[tid.get()];

    // Construct FieldTrackView
    FieldTrackView field_view(geo_view, track_view);

    // Construct the RK stepper adnd propagator in a field
    MagField                            field({0, 0, test.field_value});
    MagFieldEquation                    equation(field, field_view.charge());
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);
    FieldPropagator                     propagator(field_params, driver);

    // Tests with input parameters of a electron in a uniform magnetic field
    double hstep = (2.0 * constants::pi * test.radius) / test.nsteps;

    real_type curved_length = 0;

    constexpr int num_boundary = 16;
    int           icross       = 0;

    // clang-format off
    constexpr real_type expected_y[num_boundary]
        = { 0.5,  1.5,  2.5,  3.5,  3.5,  2.5,  1.5,  0.5,
           -0.5, -1.5, -2.5, -3.5, -3.5, -2.5, -1.5, -0.5};
    // clang-format on

    real_type delta = celeritas::numeric_limits<real_type>::max();
    ;

    for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
    {
        for (CELER_MAYBE_UNUSED int i : celeritas::range(test.nsteps))
        {
            field_view.step() = hstep;
            curved_length += propagator(&field_view);

            if (field_view.on_boundary())
            {
                icross++;
                int j = (icross - 1) % num_boundary;
                delta = expected_y[j] - field_view.state().pos[1];
                if (delta != 0)
                {
                    printf("Intersection Finding Failed: ");
                    printf("Expected = %f Actual = %f\n",
                           expected_y[j],
                           field_view.state().pos[1]);
                }
            }
        }
    }

    // output
    step[tid.get()] = curved_length;
    pos[tid.get()]  = field_view.state().pos[0];
    mom[tid.get()]  = field_view.state().mom[1];
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

    // Temporary device data for kernel
    thrust::device_vector<VGGTestInit>        in_geo(input.init_geo.begin(),
                                              input.init_geo.end());
    thrust::device_vector<ParticleTrackState> in_track = input.init_track;

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
        input.test,
        raw_pointer_cast(in_track.data()),
        raw_pointer_cast(pos.data()),
        raw_pointer_cast(mom.data()),
        raw_pointer_cast(step.data()));
    CELER_CUDA_CHECK_ERROR();
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

//! Run a boundary crossing test on device and return results

FPTestOutput bc_test(FPTestInput input)
{
    CELER_ASSERT(input.init_geo.size() == input.init_track.size());
    CELER_ASSERT(input.geo_params);
    CELER_ASSERT(input.geo_states);

    // Temporary device data for kernel
    thrust::device_vector<VGGTestInit>        in_geo(input.init_geo.begin(),
                                              input.init_geo.end());
    thrust::device_vector<ParticleTrackState> in_track = input.init_track;

    // Output data for kernel
    thrust::device_vector<double> step(input.init_geo.size(), -1.0);
    thrust::device_vector<double> pos(input.init_geo.size(), -1.0);
    thrust::device_vector<double> mom(input.init_geo.size(), -1.0);

    // Run kernel
    celeritas::KernelParamCalculator calc_launch_params(bc_test_kernel,
                                                        "bc_test");
    auto params = calc_launch_params(in_geo.size());

    bc_test_kernel<<<params.grid_size, params.block_size>>>(
        in_geo.size(),
        input.geo_params,
        input.geo_states,
        raw_pointer_cast(in_geo.data()),
        input.particle_params,
        input.particle_states,
        input.field_params,
        input.test,
        raw_pointer_cast(in_track.data()),
        raw_pointer_cast(pos.data()),
        raw_pointer_cast(mom.data()),
        raw_pointer_cast(step.data()));
    CELER_CUDA_CHECK_ERROR();
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
