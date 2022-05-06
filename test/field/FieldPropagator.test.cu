//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file field/FieldPropagator.test.cu
//---------------------------------------------------------------------------//
#include "FieldPropagator.test.hh"

#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldParamsData.hh"
#include "celeritas/field/FieldPropagator.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MagFieldTraits.hh"
#include "celeritas/field/UniformMagField.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "FieldTestParams.hh"

using namespace celeritas;
using thrust::raw_pointer_cast;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void fp_test_kernel(const int                       size,
                               const GeoParamsCRefDevice       shared,
                               const GeoStateRefDevice         state,
                               const GeoTrackInitializer*      start,
                               const ParticleParamsRef         particle_params,
                               ParticleStateRef                particle_states,
                               FieldParamsData                 field_params,
                               FieldTestParams                 test,
                               const ParticleTrackInitializer* init_track,
                               double*                         pos,
                               double*                         dir,
                               double*                         step)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize GeoTrackView and ParticleTrackView
    GeoTrackView geo_track(shared, state, tid);
    geo_track = start[tid.get()];
    if (!geo_track.is_outside())
        geo_track.find_next_step();

    ParticleTrackView particle_track(particle_params, particle_states, tid);
    particle_track = init_track[tid.get()];

    // Construct the field propagator with UniformMagField
    UniformMagField field({0, 0, test.field_value});
    using MFTraits = MagFieldTraits<UniformMagField, DormandPrinceStepper>;
    MFTraits::Equation_t   equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t    stepper(equation);
    MFTraits::Driver_t     driver(field_params, &stepper);
    MFTraits::Propagator_t propagator(particle_track, &geo_track, &driver);

    // Tests with input parameters of a electron in a uniform magnetic field
    double hstep = (2.0 * constants::pi * test.radius) / test.nsteps;

    real_type curved_length = 0;

    MFTraits::Propagator_t::result_type result;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test.revolutions))
    {
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test.nsteps))
        {
            result = propagator(hstep);
            curved_length += result.distance;
            CELER_ASSERT(!result.boundary);
        }
    }

    // output
    step[tid.get()] = curved_length;
    pos[tid.get()]  = geo_track.pos()[0];
    dir[tid.get()]  = geo_track.dir()[1];
}

__global__ void bc_test_kernel(const int                       size,
                               const GeoParamsCRefDevice       shared,
                               const GeoStateRefDevice         state,
                               const GeoTrackInitializer*      start,
                               ParticleParamsRef               particle_params,
                               ParticleStateRef                particle_states,
                               FieldParamsData                 field_params,
                               FieldTestParams                 test,
                               const ParticleTrackInitializer* init_track,
                               double*                         pos,
                               double*                         dir,
                               double*                         step)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= size)
        return;

    // Initialize GeoTrackView and ParticleTrackView
    GeoTrackView geo_track(shared, state, tid);
    geo_track = start[tid.get()];
    if (!geo_track.is_outside())
        geo_track.find_next_step();

    ParticleTrackView particle_track(particle_params, particle_states, tid);
    particle_track = init_track[tid.get()];

    // Construct the field propagator with UniformMagField
    UniformMagField field({0, 0, test.field_value});
    using MFTraits = MagFieldTraits<UniformMagField, DormandPrinceStepper>;
    MFTraits::Equation_t   equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t    stepper(equation);
    MFTraits::Driver_t     driver(field_params, &stepper);
    MFTraits::Propagator_t propagator(particle_track, &geo_track, &driver);

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

    MFTraits::Propagator_t::result_type result;

    for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
    {
        for (CELER_MAYBE_UNUSED int i : celeritas::range(test.nsteps))
        {
            result = propagator(hstep);
            curved_length += result.distance;

            if (result.boundary)
            {
                icross++;
                int j = (icross - 1) % num_boundary;
                delta = expected_y[j] - geo_track.pos()[1];
                if (delta != 0)
                {
                    printf("Intersection Finding Failed on GPU: ");
                    printf("Expected = %f Actual = %f\n",
                           expected_y[j],
                           geo_track.pos()[1]);
                }
                geo_track.cross_boundary();
            }
        }
    }

    // output
    step[tid.get()] = curved_length;
    pos[tid.get()]  = geo_track.pos()[0];
    Real3 final_dir = geo_track.dir();
    normalize_direction(&final_dir);
    dir[tid.get()] = final_dir[1];
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
    thrust::device_vector<GeoTrackInitializer> in_geo(input.init_geo.begin(),
                                                      input.init_geo.end());
    thrust::device_vector<ParticleTrackInitializer> in_track = input.init_track;

    // Output data for kernel
    thrust::device_vector<double> step(input.init_geo.size(), -1.0);
    thrust::device_vector<double> pos(input.init_geo.size(), -1.0);
    thrust::device_vector<double> dir(input.init_geo.size(), -1.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(fp_test,
                        celeritas::device().default_block_size(),
                        in_geo.size(),
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
                        raw_pointer_cast(dir.data()),
                        raw_pointer_cast(step.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy result back to CPU
    FPTestOutput result;

    result.step.resize(step.size());
    thrust::copy(step.begin(), step.end(), result.step.begin());

    result.pos.resize(pos.size());
    thrust::copy(pos.begin(), pos.end(), result.pos.begin());

    result.dir.resize(dir.size());
    thrust::copy(dir.begin(), dir.end(), result.dir.begin());

    return result;
}

//! Run a boundary crossing test on device and return results

FPTestOutput bc_test(FPTestInput input)
{
    CELER_ASSERT(input.init_geo.size() == input.init_track.size());
    CELER_ASSERT(input.geo_params);
    CELER_ASSERT(input.geo_states);

    // Temporary device data for kernel
    thrust::device_vector<GeoTrackInitializer> in_geo(input.init_geo.begin(),
                                                      input.init_geo.end());
    thrust::device_vector<ParticleTrackInitializer> in_track = input.init_track;

    // Output data for kernel
    thrust::device_vector<double> step(input.init_geo.size(), -1.0);
    thrust::device_vector<double> pos(input.init_geo.size(), -1.0);
    thrust::device_vector<double> dir(input.init_geo.size(), -1.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(bc_test,
                        celeritas::device().default_block_size(),
                        in_geo.size(),
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
                        raw_pointer_cast(dir.data()),
                        raw_pointer_cast(step.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy result back to CPU
    FPTestOutput result;

    result.step.resize(step.size());
    thrust::copy(step.begin(), step.end(), result.step.begin());

    result.pos.resize(pos.size());
    thrust::copy(pos.begin(), pos.end(), result.pos.begin());

    result.dir.resize(dir.size());
    thrust::copy(dir.begin(), dir.end(), result.dir.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
