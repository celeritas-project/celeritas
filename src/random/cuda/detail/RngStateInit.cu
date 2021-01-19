//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateInit.cu
//---------------------------------------------------------------------------//
#include "RngStateInit.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/cuda/RngEngine.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
__global__ void rng_init_kernel(const RngStatePointers           state,
                                const RngSeed::value_type* const seeds)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < state.size())
    {
        RngEngine rng(state, tid);
        rng = RngEngine::Initializer_t{seeds[tid.get()]};
    }
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
void rng_state_init_device(const RngStatePointers&         device_ptrs,
                           Span<const RngSeed::value_type> device_seeds)
{
    CELER_EXPECT(device_ptrs.size() == device_seeds.size());

    // Launch kernel to build RNG states on device
    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(device_seeds.size());
    rng_init_kernel<<<params.grid_size, params.block_size>>>(
        device_ptrs, device_seeds.data());
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
