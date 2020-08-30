//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateInit.cu
//---------------------------------------------------------------------------//
#include "RngStateInit.hh"

#include "base/KernelParamCalculator.cuda.hh"
#include "RngEngine.cuh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
__global__ void
init_impl(RngStatePointers const state, const RngSeed::value_type* const seeds)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < state.rng.size())
    {
        RngEngine rng(state, tid);
        rng = RngEngine::Initializer_t{seeds[tid.get()]};
    }
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
void rng_state_init_device(const RngStatePointers&         device_ptrs,
                           span<const RngSeed::value_type> device_seeds)
{
    REQUIRE(device_ptrs.rng.size() == device_seeds.size());

    // Launch kernel to build RNG states on device
    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(device_seeds.size());
    init_impl<<<params.grid_size, params.block_size>>>(device_ptrs,
                                                       device_seeds.data());
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
