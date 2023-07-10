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

__global__ void dormand_prince_test_kernel()
{
    auto eval = make_dummy_equation(dormand_prince_dummy_field);
    auto stepper = DormandPrinceStepper{eval};
    OdeState state = eval({{1, 2, 3}, {0, 0, 1}});
    FieldStepperResult res;

    for (int i = 0; i < 100; ++i)
    {
        res = stepper(1, state);
        state = res.mid_state;
    }
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void dormand_prince_cuda_test()
{
    CELER_LOG(info) << "Running Dormand-Prince stepper test on CUDA";
    DormandPrinceKernelArguments params{100, FieldStepperResult()};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel with the desired streamId
    ScopedProfiling profile_this{"Dormand-Prince-test"};
    
    CELER_LAUNCH_KERNEL(dormand_prince_test, device().default_block_size(), 1, 0);

    // Wait for the kernel with the desired streamId to finish executing
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Print the result
    print_result(params.result);

    // Compute the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    // Print the elapsed time
    CELER_LOG(info) << "Kernel runtime: " << milliseconds << " ms";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
