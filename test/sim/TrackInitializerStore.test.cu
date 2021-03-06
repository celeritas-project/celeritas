//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitializerStore.test.cu
//---------------------------------------------------------------------------//
#include "TrackInitializerStore.test.hh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"

namespace celeritas_test
{
using namespace celeritas;

//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void interact_kernel(StatePointers              states,
                                SecondaryAllocatorPointers secondaries,
                                ITTestInputPointers        input)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        SimTrackView sim(states.sim, thread_id);

        // There may be more track slots than active tracks; only active tracks
        // should interact
        if (sim.alive())
        {
            // Allow the particle to interact and create secondaries
            StackAllocator<Secondary> allocate_secondaries(secondaries);
            Interactor             interact(allocate_secondaries,
                                input.alloc_size[thread_id.get()],
                                input.alive[thread_id.get()]);
            states.interactions[thread_id.get()] = interact();

            // Kill the selected tracks
            if (!input.alive[thread_id.get()])
            {
                sim.alive() = false;
            }
        }
        else
        {
            states.interactions[thread_id.get()]
                = Interaction::from_absorption();
        }
    }
}

__global__ void tracks_test_kernel(StatePointers states, unsigned int* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();
    if (thread_id < states.size())
    {
        SimTrackView sim(states.sim, thread_id);
        output[thread_id.get()] = sim.track_id().get();
    }
}

__global__ void
initializers_test_kernel(TrackInitializerPointers inits, unsigned int* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();
    if (thread_id < inits.initializers.size())
    {
        TrackInitializer& init  = inits.initializers[thread_id.get()];
        output[thread_id.get()] = init.sim.track_id.get();
    }
}

__global__ void
vacancies_test_kernel(TrackInitializerPointers inits, size_type* output)
{
    auto thread_id = celeritas::KernelParamCalculator::thread_id();
    if (thread_id < inits.vacancies.size())
    {
        output[thread_id.get()] = inits.vacancies[thread_id.get()];
    }
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

void interact(StatePointers              states,
              SecondaryAllocatorPointers secondaries,
              ITTestInputPointers        input)
{
    CELER_EXPECT(states.size() > 0);
    CELER_EXPECT(states.size() == input.alloc_size.size());

    static const KernelParamCalculator calc_launch_params(interact_kernel,
                                                          "interact");
    auto                  lparams = calc_launch_params(states.size());
    interact_kernel<<<lparams.grid_size, lparams.block_size>>>(
        states, secondaries, input);
    CELER_CUDA_CHECK_ERROR();
}

std::vector<unsigned int> tracks_test(StatePointers states)
{
    // Allocate memory for results
    std::vector<unsigned int> host_output(states.size());
    if (states.size() == 0)
    {
        return host_output;
    }
    thrust::device_vector<unsigned int> output(states.size());

    // Launch a kernel to check the track ID of the initialized tracks
    static const celeritas::KernelParamCalculator calc_launch_params(
        tracks_test_kernel, "tracks_test");
    auto                  lparams = calc_launch_params(states.size());
    tracks_test_kernel<<<lparams.grid_size, lparams.block_size>>>(
        states, thrust::raw_pointer_cast(output.data()));
    CELER_CUDA_CHECK_ERROR();

    // Copy data back to host
    thrust::copy(output.begin(), output.end(), host_output.begin());

    return host_output;
}

std::vector<unsigned int> initializers_test(TrackInitializerPointers inits)
{
    // Allocate memory for results
    std::vector<unsigned int> host_output(inits.initializers.size());
    if (inits.initializers.size() == 0)
    {
        return host_output;
    }
    thrust::device_vector<unsigned int> output(inits.initializers.size());

    // Launch a kernel to check the track ID of the track initializers
    static const celeritas::KernelParamCalculator calc_launch_params(
        initializers_test_kernel, "initializers_test");
    auto lparams = calc_launch_params(inits.initializers.size());
    initializers_test_kernel<<<lparams.grid_size, lparams.block_size>>>(
        inits, thrust::raw_pointer_cast(output.data()));
    CELER_CUDA_CHECK_ERROR();

    // Copy data back to host
    thrust::copy(output.begin(), output.end(), host_output.begin());

    return host_output;
}

std::vector<size_type> vacancies_test(TrackInitializerPointers inits)
{
    // Allocate memory for results
    std::vector<size_type> host_output(inits.vacancies.size());
    if (inits.vacancies.size() == 0)
    {
        return host_output;
    }
    thrust::device_vector<size_type> output(inits.vacancies.size());

    // Launch a kernel to check the indices of the empty slots
    static const celeritas::KernelParamCalculator calc_launch_params(
        vacancies_test_kernel, "vacancies_test");
    auto                  lparams = calc_launch_params(inits.vacancies.size());
    vacancies_test_kernel<<<lparams.grid_size, lparams.block_size>>>(
        inits, thrust::raw_pointer_cast(output.data()));
    CELER_CUDA_CHECK_ERROR();

    // Copy data back to host
    thrust::copy(output.begin(), output.end(), host_output.begin());

    return host_output;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
