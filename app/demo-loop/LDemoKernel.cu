//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cu
//---------------------------------------------------------------------------//
#include "LDemoKernel.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "physics/base/PhysicsStepUtils.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void
pre_step_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    CELER_NOT_IMPLEMENTED("Pre-step kernel");
#if 0
    PhysicsTrackView phys(inp.params, inp.states, init.particle, init.mat, tid);
    phys = PhysicsTrackInitializer{};

    // Sample mean free path
    {
        RngEngine                 rng(states.rng, ThreadId(tid));
        ExponentialDistribution<> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }
#endif
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
void pre_step(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(pre_step, states.size(), params, states);
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
