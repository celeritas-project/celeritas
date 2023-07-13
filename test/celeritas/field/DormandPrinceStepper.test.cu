//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DormandPrinceStepper.cu
//---------------------------------------------------------------------------//
#include "DormandPrinceStepper.test.hh"

#include "corecel/sys/KernelParamCalculator.device.hh" // for CELER_LAUNCH_KERNEL
#include "corecel/sys/Device.hh" // device()
#include "corecel/sys/ScopedProfiling.hh" // ScopedProfiling

#include <typeinfo>

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
__global__ void dormand_test_arg_kernel(OdeState *states,
                                        FieldStepperResult *results,
                                        int *num_states, int *number_iterations)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < *num_states)
    {
        // printf("Executing thread %d\n", index);
        auto eval = make_dummy_equation(dormand_prince_dummy_field);
        Stepper_impl stepper{eval};
        FieldStepperResult res;
        auto state = states[index];
        for (int i = 0; i < *number_iterations; ++i)
        {
            res = stepper(1, state);
            state = res.mid_state;
        }
        results[index] = stepper(1, states[index]);

        // printf("Finished thread %d\n", index);
    }
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
KernelResult simulate_multi_next_chord()
{
    KernelResult result;

    // Load initial states and results to device
    int *d_num_states, *d_number_iterations;
    FieldStepperResult *h_results, *d_results;
    OdeState *d_states;

    h_results = new FieldStepperResult[number_of_states];
    for (int i = 0; i < number_of_states; ++i)
    {
        h_results[i] = FieldStepperResult();
    }

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on device
    cudaMalloc(&d_results, number_of_states * sizeof(FieldStepperResult));
    cudaMalloc(&d_states, number_of_states * sizeof(OdeState));
    cudaMalloc(&d_num_states, sizeof(int));
    cudaMalloc(&d_number_iterations, sizeof(int));

    // Copy initial states to device
    cudaMemcpy(d_states, initial_states, number_of_states * sizeof(OdeState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_states, &number_of_states, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_iterations, &number_iterations, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with the desired streamId
    // ScopedProfiling profile_this{"Dormand-Prince-test"};
    cudaEventRecord(start);
    // CELER_LAUNCH_KERNEL(dormand_test_arg,
    //                     device().threads_per_warp(), number_of_states, 0,
    //                     d_states, d_results, d_num_states);
    dormand_test_arg_kernel<Stepper_DormandPrince><<<1, number_of_states>>>
        (d_states, d_results, d_num_states, d_number_iterations);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // Compute the elapsed time
    cudaEventElapsedTime(&(result.milliseconds), start, stop);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_results, d_results, number_of_states * sizeof(FieldStepperResult), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_num_states);

    // Return results
    result.results = h_results;
    return result;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
