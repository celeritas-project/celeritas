//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UserParamField.test.cu
//---------------------------------------------------------------------------//
#include <thrust/device_vector.h>

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/Constants.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldParamsData.hh"
#include "celeritas/field/FieldPropagator.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MagFieldTraits.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "FieldPropagator.test.hh"
#include "FieldTestParams.hh"
#include "UserField.test.hh"
#include "detail/CMSParameterizedField.hh"

using celeritas_test::detail::CMSParameterizedField;
using thrust::raw_pointer_cast;

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void parameterized_field_test_kernel(UserFieldTestParams param,
                                                real_type*          value_x,
                                                real_type*          value_y,
                                                real_type*          value_z)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= param.nsamples)
        return;

    detail::CMSParameterizedField field;
    Real3                         pos{tid.get() * param.delta_r,
              tid.get() * param.delta_r,
              tid.get() * param.delta_z};
    Real3                         value = field(pos);

    // Output for verification
    value_x[tid.get()] = value[0];
    value_y[tid.get()] = value[1];
    value_z[tid.get()] = value[2];
}

__global__ void par_fp_test_kernel(const int                  size,
                                   const GeoParamsCRefDevice  shared,
                                   const GeoStateRefDevice    state,
                                   const GeoTrackInitializer* start,
                                   const ParticleParamsRef    particle_params,
                                   ParticleStateRef           particle_states,
                                   FieldParamsData            field_params,
                                   FieldTestParams            test,
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

    // Construct the field propagator with a user CMSParameterizedField
    CMSParameterizedField field;
    using MFTraits
        = MagFieldTraits<CMSParameterizedField, DormandPrinceStepper>;
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
}

__global__ void par_bc_test_kernel(const int                  size,
                                   const GeoParamsCRefDevice  shared,
                                   const GeoStateRefDevice    state,
                                   const GeoTrackInitializer* start,
                                   ParticleParamsRef          particle_params,
                                   ParticleStateRef           particle_states,
                                   FieldParamsData            field_params,
                                   FieldTestParams            test,
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

    // Construct the field propagator with a user CMSParameterizedField
    CMSParameterizedField field;
    using MFTraits
        = MagFieldTraits<CMSParameterizedField, DormandPrinceStepper>;
    MFTraits::Equation_t   equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t    stepper(equation);
    MFTraits::Driver_t     driver(field_params, &stepper);
    MFTraits::Propagator_t propagator(particle_track, &geo_track, &driver);

    // Tests with input parameters of a electron in a uniform magnetic field
    double hstep = (2.0 * constants::pi * test.radius) / test.nsteps;

    real_type curved_length = 0;

    constexpr int num_boundary = 4;
    int           icross       = 0;

    constexpr real_type expected_y[num_boundary] = {0.5, 0.5, -0.5, -0.5};

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
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
UserFieldTestOutput parameterized_field_test(UserFieldTestParams test_param)
{
    // Output data for kernel
    thrust::device_vector<real_type> value_x(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_y(test_param.nsamples, 0.0);
    thrust::device_vector<real_type> value_z(test_param.nsamples, 0.0);

    // Run kernel
    CELER_LAUNCH_KERNEL(parameterized_field_test,
                        celeritas::device().default_block_size(),
                        test_param.nsamples,
                        test_param,
                        raw_pointer_cast(value_x.data()),
                        raw_pointer_cast(value_y.data()),
                        raw_pointer_cast(value_z.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy result back to CPU
    UserFieldTestOutput result;

    result.value_x.resize(value_x.size());
    thrust::copy(value_x.begin(), value_x.end(), result.value_x.begin());

    result.value_y.resize(value_y.size());
    thrust::copy(value_y.begin(), value_y.end(), result.value_y.begin());

    result.value_z.resize(value_z.size());
    thrust::copy(value_z.begin(), value_z.end(), result.value_z.begin());

    return result;
}

//! Run on device and return results
UserFieldTestVector par_fp_test(FPTestInput input)
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
    CELER_LAUNCH_KERNEL(par_fp_test,
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
    UserFieldTestVector result;

    result.resize(step.size());
    thrust::copy(step.begin(), step.end(), result.begin());

    return result;
}

//! Run a boundary crossing test on device and return results
UserFieldTestVector par_bc_test(FPTestInput input)
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
    CELER_LAUNCH_KERNEL(par_bc_test,
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
    UserFieldTestVector result;

    result.resize(step.size());
    thrust::copy(step.begin(), step.end(), result.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
