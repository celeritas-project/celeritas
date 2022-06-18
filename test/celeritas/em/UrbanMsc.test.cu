//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cu
//---------------------------------------------------------------------------//
#include "UrbanMsc.test.hh"

#include <thrust/device_vector.h>

#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/em/distribution/UrbanMscHelper.hh"
#include "celeritas/em/distribution/UrbanMscScatter.hh"
#include "celeritas/em/distribution/UrbanMscStepLimit.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/random/RngEngine.hh"

using thrust::raw_pointer_cast;
using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void msc_test_kernel(MscTestInput                    input,
                                const ParticleTrackInitializer* init_part,
                                const PhysTestInit*             init_phys,
                                MscTestOutput*                  output)
{
    auto         tid = celeritas::KernelParamCalculator::thread_id();
    unsigned int idx = tid.get();

    if (idx >= input.test_param.nstates)
        return;

    // Create and initialize views
    GeoTrackView geo_view(input.geometry_params, input.geometry_states, tid);
    geo_view = {input.test_param.position, input.test_param.direction};

    ParticleTrackView particle_view(
        input.particle_params, input.particle_states, tid);
    particle_view = init_part[idx];

    PhysicsTrackView phys_view(input.physics_params,
                               input.physics_states,
                               init_phys[idx].particle,
                               init_phys[idx].material,
                               tid);
    phys_view = PhysicsTrackInitializer{};

    MaterialView mat_view(input.material_params, init_phys[idx].material);

    // Calculate the multiple scattering step limitation
    UrbanMscHelper msc_helper(input.msc_data, particle_view, phys_view);

    UrbanMscStepLimit step_limiter(input.msc_data,
                                   particle_view,
                                   phys_view,
                                   init_phys[idx].material,
                                   true,
                                   geo_view.find_safety(),
                                   msc_helper.range());

    RngEngine rng_engine(input.rng_states, tid);
    MscStep   step_result = step_limiter(rng_engine);

    // Propagate up to the geometric step length
    real_type        geo_step = step_result.geom_path;
    LinearPropagator propagate(&geo_view);
    auto             propagated = propagate(geo_step);

    if (propagated.boundary)
    {
        // Stopped at a geometry boundary:
        step_result.geom_path = propagated.distance;
    }

    // Sample the multiple scattering
    UrbanMscScatter scatter(input.msc_data,
                            particle_view,
                            &geo_view,
                            phys_view,
                            mat_view,
                            step_result);

    MscInteraction sample_result = scatter(rng_engine);

    // Output for physics verification
    output[idx] = celeritas_test::calc_output(step_result, sample_result);
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run the UrbanMsc test on device and return results
std::vector<MscTestOutput> msc_test(MscTestInput input)
{
    CELER_ASSERT(input.init_part.size() == input.particle_states.size());
    CELER_ASSERT(input.init_phys.size() == input.physics_states.size());

    // Temporary device data for kernel
    thrust::device_vector<ParticleTrackInitializer> in_part = input.init_part;
    thrust::device_vector<PhysTestInit>             in_phys = input.init_phys;

    // Output data for kernel
    thrust::device_vector<MscTestOutput> output(input.test_param.nstates);

    // Run kernel
    CELER_LAUNCH_KERNEL(msc_test,
                        celeritas::device().default_block_size(),
                        input.test_param.nstates,
                        input,
                        raw_pointer_cast(in_part.data()),
                        raw_pointer_cast(in_phys.data()),
                        raw_pointer_cast(output.data()));
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());

    // Copy result back to CPU
    std::vector<MscTestOutput> result;
    result.resize(output.size());
    thrust::copy(output.begin(), output.end(), result.begin());

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
