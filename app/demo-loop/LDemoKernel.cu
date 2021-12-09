//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cu
//---------------------------------------------------------------------------//
#include "LDemoKernel.hh"

#include "base/KernelParamCalculator.cuda.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Sample mean free path and calculate physics step limits.
 */
__global__ void
pre_step_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    PreStepLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate and process physical changes to the track along the step and
 * select the process/model for discrete interaction.
 */
__global__ void along_and_post_step_kernel(ParamsDeviceRef const params,
                                           StateDeviceRef const  states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    AlongAndPostStepLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
__global__ void process_interactions_kernel(ParamsDeviceRef const params,
                                            StateDeviceRef const  states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    ProcessInteractionsLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
__global__ void
cleanup_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    CleanupLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}

} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
#define CDL_LAUNCH_KERNEL(NAME, THREADS, ARGS...)                   \
    do                                                              \
    {                                                               \
        static const ::celeritas::KernelParamCalculator NAME##_ckp( \
            NAME##_kernel, #NAME);                                  \
        auto kp = NAME##_ckp(THREADS);                              \
                                                                    \
        NAME##_kernel<<<kp.grid_size, kp.block_size>>>(ARGS);       \
        CELER_CUDA_CHECK_ERROR();                                   \
    } while (0)

//---------------------------------------------------------------------------//
/*!
 * Get minimum step length from interactions.
 */
void pre_step(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(pre_step, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Propogation, slowing down, and discrete model selection.
 */
void along_and_post_step(const ParamsDeviceRef& params,
                         const StateDeviceRef&  states)
{
    CDL_LAUNCH_KERNEL(along_and_post_step, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
void process_interactions(const ParamsDeviceRef& params,
                          const StateDeviceRef&  states)
{
    CDL_LAUNCH_KERNEL(process_interactions, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
void cleanup(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(cleanup, 1, params, states);
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
