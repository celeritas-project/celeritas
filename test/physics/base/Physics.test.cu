//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Physics.test.cu
//---------------------------------------------------------------------------//
#include "Physics.test.hh"

#include "base/KernelParamCalculator.cuda.hh"

using namespace celeritas;

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void phys_test_kernel(const PTestInput inp)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= inp.states.size())
        return;

    const auto&      init = inp.inits[tid];
    PhysicsTrackView phys(inp.params, inp.states, init.particle, init.mat, tid);

    phys                  = PhysicsTrackInitializer{};
    inp.result[tid.get()] = calc_step(phys, init.energy);
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void phys_cuda_test(const PTestInput& input)
{
    CELER_ASSERT(input.inits.size() == input.states.size());
    static const KernelParamCalculator calc_launch_params(phys_test_kernel,
                                                          "phys_test");
    auto params = calc_launch_params(input.states.size());
    phys_test_kernel<<<params.grid_size, params.block_size>>>(input);

    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
