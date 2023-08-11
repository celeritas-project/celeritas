//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cu
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include <typeinfo>

#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/field/detail/FieldUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
template<class Stepper_impl>
__device__ FieldStepperResult run_stepper(Stepper_impl& stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          int number_states,
                                          OdeState* ks,
                                          OdeState* along_states,
                                          FieldStepperResult* result);

template<>
__device__ FieldStepperResult run_stepper(StepperUni& stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          int number_states,
                                          OdeState* ks,
                                          OdeState* along_states,
                                          FieldStepperResult* result)
{
    return stepper(step, state);
}

template<>
__device__ FieldStepperResult run_stepper(StepperMultiGlobal& stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          int number_states,
                                          OdeState* ks,
                                          OdeState* along_states,
                                          FieldStepperResult* result)
{
    return stepper(step, state, number_threads, ks, along_states, result);
}

template<>
__device__ FieldStepperResult run_stepper(StepperMultiShared& stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          int number_states,
                                          OdeState* ks,
                                          OdeState* along_states,
                                          FieldStepperResult* result)
{
    return stepper(step, state, number_threads, number_states);
}

template<class Stepper_impl>
__global__ void dormand_test_arg_kernel(OdeState* states,
                                        FieldStepperResult* results,
                                        int number_iterations,
                                        int number_threads,
                                        int number_states,
                                        OdeState* ks,
                                        OdeState* along_states)
{
    constexpr double initial_step_size = 10000.0;
    constexpr double delta_chord = 1e-4;
    constexpr double half = 0.5;

    auto id = KernelParamCalculator::thread_id().get() / number_threads;
    if (id >= number_states)
    {
        return;
    }

    auto eval = make_dummy_equation(dormand_prince_dummy_field);
    Stepper_impl stepper{eval};
    FieldStepperResult res;
    auto state = states[id];
    auto step = initial_step_size;

    for (int i = 0; i < number_iterations; ++i)
    {
        res = run_stepper(stepper,
                          step,
                          state,
                          number_threads,
                          number_states,
                          &ks[id * 7],
                          &along_states[id],
                          &results[id]);
        auto dchord
            = detail::distance_chord(state, res.mid_state, res.end_state);
        step *= max(std::sqrt(delta_chord / dchord), half);
    }
    results[id] = res;
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
KernelResult simulate_multi_next_chord(int number_threads,
                                       int number_states,
                                       bool use_shared)
{
    KernelResult result;
    bool is_global_version = number_threads > 1 && !use_shared;

    // Load initial states and results to device
    FieldStepperResult* d_results;
    std::vector<FieldStepperResult> h_results(number_states);

    OdeState* d_along_states;
    std::vector<OdeState> h_along_states(number_states);

    //TODO: Move this into "build_variables"
    OdeState *h_ks, *d_ks;
    if (is_global_version)
    {
        h_ks = new OdeState[number_states * 7];
        for (int i = 0; i < number_states * 7; ++i)
        {
            h_ks[i] = OdeState();
        }
    }

    OdeState *h_states, *d_states;
    h_states = new OdeState[number_states];

    build_variables(number_states, is_global_version,
    h_results.data(), h_along_states.data(), h_states);

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_results, number_states * sizeof(FieldStepperResult));
    cudaMalloc(&d_states, number_states * sizeof(OdeState));
    if (is_global_version)
    {
        cudaMalloc(&d_ks, number_states * 7 * sizeof(OdeState));
        cudaMalloc(&d_along_states, number_states * sizeof(OdeState));
    }

    // Copy initial states to device
    cudaMemcpy(d_states,
               h_states,
               number_states * sizeof(OdeState),
               cudaMemcpyHostToDevice);
    if (is_global_version)
    {
        cudaMemcpy(d_ks,
                   h_ks,
                   number_states * 7 * sizeof(OdeState),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_along_states,
                   h_along_states.data(),
                   number_states * sizeof(OdeState),
                   cudaMemcpyHostToDevice);
    }

    // Kernel configuration
    constexpr int max_threads_per_block = 768;
    int grid_size = number_threads * number_states / max_threads_per_block;
    grid_size += (number_threads * number_states) % max_threads_per_block
                    == 0
        ? 0
        : 1;
    int nb_warps = number_threads * number_states / (32 * grid_size);
    nb_warps += (number_threads * number_states) % (32 * grid_size) == 0 ? 0 : 1;
    int block_size = nb_warps * 32;
    int shared_memory = 0;
    if (use_shared)
    {
        constexpr int max_threads_per_block_shared = 320;
        grid_size = number_threads * number_states / max_threads_per_block_shared;
        grid_size += (number_threads * number_states) % max_threads_per_block_shared
                        == 0
            ? 0
            : 1;
        nb_warps = number_threads * number_states / (32 * grid_size);
        nb_warps += (number_threads * number_states) % (32 * grid_size) == 0 ? 0 : 1;
        block_size = nb_warps * 32;

        shared_memory = (block_size / 4) * 7 * sizeof(OdeState)
                        + (block_size / 4) * sizeof(OdeState)
                        + (block_size / 4) * sizeof(FieldStepperResult);
    }

    // Launch the kernel with the desired streamId
    cudaEventRecord(start);
    if (number_threads > 1)
    {
        if (use_shared)
        {
            dormand_test_arg_kernel<StepperMultiShared>
                <<<grid_size, block_size, shared_memory>>>(
                    d_states,
                    d_results,
                    number_iterations,
                    number_threads,
                    number_states,
                    d_ks,
                    d_along_states);
        }
        else
        {
            dormand_test_arg_kernel<StepperMultiGlobal>
                <<<grid_size, block_size>>>(d_states,
                                                        d_results,
                                                        number_iterations,
                                                        number_threads,
                                                        number_states,
                                                        d_ks,
                                                        d_along_states);
        }
    }
    else
    {
        dormand_test_arg_kernel<StepperUni>
            <<<grid_size, block_size>>>(d_states,
                                                    d_results,
                                                    number_iterations,
                                                    number_threads,
                                                        number_states,
                                                    d_ks,
                                                    d_along_states);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // Check if kernel execution generated an error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        int max_threads_per_block = 0;
        int max_blocks = 0;
        int max_shared_memory = 0;

        cudaDeviceGetAttribute(
            &max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
        cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMaxGridDimX, 0);
        cudaDeviceGetAttribute(
            &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, 0);

        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "Launch config for " << number_states
                  << " states: " << grid_size << " blocks, "
                  << block_size << " threads per block, "
                  << shared_memory << " bytes of "
                  << "shared memory"<< std::endl;
        std::cerr << "Device properties: " << max_threads_per_block
                  << " threads per block, " << max_blocks << " blocks and "
                  << max_shared_memory << " bytes of shared memory"
                  << std::endl;
        result.milliseconds = -1;
    }
    else
    {
        // Compute the elapsed time
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&(result.milliseconds), start, stop);
    }

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to hostComp
    cudaMemcpy(h_results.data(),
               d_results,
               number_states * sizeof(FieldStepperResult),
               cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_results);
    cudaFree(d_states);
    if (is_global_version)
    {
        cudaFree(d_ks);
        cudaFree(d_along_states);
    }

    // Return results
    result.results = h_results.data();
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
