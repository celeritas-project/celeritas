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
__device__ FieldStepperResult run_stepper(Stepper_impl stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          OdeState* ks,
                                          OdeState* along_state,
                                          FieldStepperResult* result)
{
    return FieldStepperResult();
}

template<>
__device__ FieldStepperResult run_stepper(Stepper_uni stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          OdeState* ks,
                                          OdeState* along_state,
                                          FieldStepperResult* result)
{
    return stepper(step, state);
}

template<>
__device__ FieldStepperResult run_stepper(Stepper_multi_global stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          OdeState* ks,
                                          OdeState* along_state,
                                          FieldStepperResult* result)
{
    return stepper(step, state, number_threads, ks, along_state, result);
}

template<>
__device__ FieldStepperResult run_stepper(Stepper_multi_shared stepper,
                                          int step,
                                          OdeState state,
                                          int number_threads,
                                          OdeState* ks,
                                          OdeState* along_state,
                                          FieldStepperResult* result)
{
    return stepper(step, state, number_threads);
}

template<class Stepper_impl>
__global__ void dormand_test_arg_kernel(OdeState* states,
                                        FieldStepperResult* results,
                                        int* number_iterations,
                                        int* number_threads,
                                        OdeState* ks,
                                        OdeState* along_state)
{
    constexpr double initial_step_size = 10000.0;
    constexpr double delta_chord = 1e-4;
    constexpr double half = 0.5;

    auto id = (blockIdx.x * blockDim.x + threadIdx.x) / *number_threads;

    auto eval = make_dummy_equation(dormand_prince_dummy_field);
    Stepper_impl stepper{eval};
    FieldStepperResult res;
    auto state = states[id];
    auto step = initial_step_size;

    for (int i = 0; i < *number_iterations; ++i)
    {
        res = run_stepper(stepper,
                          step,
                          state,
                          *number_threads,
                          &ks[id * 7],
                          &along_state[id],
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
    int arguments_size = 0;

    // Load initial states and results to device
    int *d_number_iterations, *d_number_threads;
    arguments_size += sizeof(int) * 2;

    FieldStepperResult *h_results, *d_results;
    h_results = new FieldStepperResult[number_states];
    for (int i = 0; i < number_states; ++i)
    {
        h_results[i] = FieldStepperResult();
        arguments_size += sizeof(FieldStepperResult);
    }

    OdeState *h_along_state, *d_along_state;
    if (number_threads > 1 && !use_shared)
    {
        h_along_state = new OdeState[number_states];
        for (int i = 0; i < number_states; ++i)
        {
            h_along_state[i] = OdeState();
            arguments_size += sizeof(OdeState);
        }
    }

    OdeState *h_ks, *d_ks;
    if (number_threads > 1 && !use_shared)
    {
        h_ks = new OdeState[number_states * 7];
        for (int i = 0; i < number_states * 7; ++i)
        {
            h_ks[i] = OdeState();
            arguments_size += sizeof(OdeState);
        }
    }

    OdeState *h_states, *d_states;
    h_states = new OdeState[number_states];
    for (int i = 0; i < number_states; ++i)
    {
        h_states[i] = initial_states[i % number_states_sample];
        arguments_size += sizeof(OdeState);
    }

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_results, number_states * sizeof(FieldStepperResult));
    cudaMalloc(&d_states, number_states * sizeof(OdeState));
    cudaMalloc(&d_number_iterations, sizeof(int));
    cudaMalloc(&d_number_threads, sizeof(int));
    if (number_threads > 1 && !use_shared)
    {
        cudaMalloc(&d_ks, number_states * 7 * sizeof(OdeState));
        cudaMalloc(&d_along_state, number_states * sizeof(OdeState));
    }

    // Copy initial states to device
    cudaMemcpy(d_states,
               initial_states,
               number_states * sizeof(OdeState),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_iterations,
               &number_iterations,
               sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_number_threads, &number_threads, sizeof(int), cudaMemcpyHostToDevice);

    if (number_threads > 1 && !use_shared)
    {
        cudaMemcpy(d_ks,
                   h_ks,
                   number_states * 7 * sizeof(OdeState),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_along_state,
                   &h_along_state,
                   number_states * sizeof(OdeState),
                   cudaMemcpyHostToDevice);
    }

    // Kernel configuration
    int shared_memory = 0;
    if (use_shared)
    {
        shared_memory = number_states * 7 * sizeof(OdeState)
                        + number_states * sizeof(OdeState)
                        + number_states * sizeof(FieldStepperResult);
    }
    int block_dimension = 1;
    int thread_dimension = number_threads * number_states;

    // Launch the kernel with the desired streamId
    cudaEventRecord(start);
    if (number_threads > 1)
    {
        if (use_shared)
        {
            dormand_test_arg_kernel<Stepper_multi_shared>
                <<<block_dimension, thread_dimension, shared_memory>>>(
                    d_states,
                    d_results,
                    d_number_iterations,
                    d_number_threads,
                    d_ks,
                    d_along_state);
        }
        else
        {
            dormand_test_arg_kernel<Stepper_multi_global>
                <<<block_dimension, thread_dimension>>>(d_states,
                                                        d_results,
                                                        d_number_iterations,
                                                        d_number_threads,
                                                        d_ks,
                                                        d_along_state);
        }
    }
    else
    {
        dormand_test_arg_kernel<Stepper_uni>
            <<<block_dimension, thread_dimension>>>(d_states,
                                                    d_results,
                                                    d_number_iterations,
                                                    d_number_threads,
                                                    d_ks,
                                                    d_along_state);
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
        std::cerr << "Launc config for " << number_states
                  << " states: " << block_dimension << " blocks, "
                  << thread_dimension << " threads per block, "
                  << shared_memory << " bytes of "
                  << "shared memory and " << arguments_size << " bytes of "
                  << "arguments" << std::endl;
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

    // Copy results back to host
    cudaMemcpy(h_results,
               d_results,
               number_states * sizeof(FieldStepperResult),
               cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_number_iterations);
    cudaFree(d_number_threads);
    if (number_threads > 1 && !use_shared)
    {
        cudaFree(d_ks);
        cudaFree(d_along_state);
    }

    // Return results
    result.results = h_results;
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
