//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateInit.cu
//---------------------------------------------------------------------------//
#include "RngStateInit.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "random/RngEngine.hh"

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
__global__ void rng_state_init_kernel(
    RngStateData<Ownership::reference, MemSpace::device> const      state,
    RngInitData<Ownership::const_reference, MemSpace::device> const init)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < state.size())
    {
        RngEngine rng(state, tid);
        rng = init.seeds[tid];
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
void rng_state_init(
    const RngStateData<Ownership::reference, MemSpace::device>&      rng,
    const RngInitData<Ownership::const_reference, MemSpace::device>& seeds)
{
    CELER_EXPECT(rng.size() == seeds.size());

    // Launch kernel to build RNG states on device
    static const celeritas::KernelParamCalculator calc_launch_params(
        rng_state_init_kernel, "rng_state_init");
    auto params = calc_launch_params(seeds.size());
    rng_state_init_kernel<<<params.grid_size, params.block_size>>>(rng, seeds);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
