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
KernelResult simulate_multi_next_chord(int number_threads, bool use_shared)
{
    KernelResult result;

    // Load initial states and results to device
    int *d_number_iterations, *d_number_threads;

    FieldStepperResult *h_results, *d_results;
    h_results = new FieldStepperResult[number_of_states];
    for (int i = 0; i < number_of_states; ++i)
    {
        h_results[i] = FieldStepperResult();
    }

    OdeState *h_along_state, *d_along_state;
    h_along_state = new OdeState[number_of_states];
    for (int i = 0; i < number_of_states; ++i)
    {
        h_along_state[i] = OdeState();
    }

    OdeState *h_ks, *d_ks;
    h_ks = new OdeState[number_of_states * 7];
    for (int i = 0; i < number_of_states * 7; ++i)
    {
        h_ks[i] = OdeState();
    }

    OdeState* d_states;

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_results, number_of_states * sizeof(FieldStepperResult));
    cudaMalloc(&d_states, number_of_states * sizeof(OdeState));
    cudaMalloc(&d_number_iterations, sizeof(int));
    cudaMalloc(&d_number_threads, sizeof(int));
    cudaMalloc(&d_ks, number_of_states * 7 * sizeof(OdeState));
    cudaMalloc(&d_along_state, number_of_states * sizeof(OdeState));

    // Copy initial states to device
    cudaMemcpy(d_states,
               initial_states,
               number_of_states * sizeof(OdeState),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_iterations,
               &number_iterations,
               sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_number_threads, &number_threads, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks,
               h_ks,
               number_of_states * 7 * sizeof(OdeState),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_along_state,
               &h_along_state,
               number_of_states * sizeof(OdeState),
               cudaMemcpyHostToDevice);

    // Kernel configuration
    int thread_dimension = number_threads * number_of_states;
    int shared_memory = number_of_states * 7 * sizeof(OdeState)
                        + number_of_states * sizeof(OdeState)
                        + number_of_states * sizeof(FieldStepperResult);

    // Launch the kernel with the desired streamId
    cudaEventRecord(start);
    if (number_threads > 1)
    {
        if (use_shared)
        {
            dormand_test_arg_kernel<Stepper_multi_shared>
                <<<1, thread_dimension, shared_memory>>>(d_states,
                                                         d_results,
                                                         d_number_iterations,
                                                         d_number_threads,
                                                         d_ks,
                                                         d_along_state);
        }
        else
        {
            dormand_test_arg_kernel<Stepper_multi_global>
                <<<1, thread_dimension>>>(d_states,
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
            <<<1, thread_dimension>>>(d_states,
                                      d_results,
                                      d_number_iterations,
                                      d_number_threads,
                                      d_ks,
                                      d_along_state);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // Compute the elapsed time
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&(result.milliseconds), start, stop);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host
    cudaMemcpy(h_results,
               d_results,
               number_of_states * sizeof(FieldStepperResult),
               cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_number_iterations);
    cudaFree(d_number_threads);
    cudaFree(d_ks);
    cudaFree(d_along_state);

    // Return results
    result.results = h_results;
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
