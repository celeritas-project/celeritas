//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStateInit.cu
//---------------------------------------------------------------------------//
#include "SimStateInit.hh"

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "../SimTrackView.hh"

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
 * Initialize the sim states on device (setting 'alive' to false).
 */
__global__ void sim_init_kernel(const SimStatePointers state)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < state.size())
    {
        SimTrackView sim_view(state, tid);
        sim_view = SimTrackView::Initializer_t{};
    }
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Initialize the sim states on device.
 */
void sim_state_init_device(const SimStatePointers& device_ptrs)
{
    // Launch kernel to build sim states on device
    static const celeritas::KernelParamCalculator calc_launch_params(
        sim_init_kernel, "sim_init");
    auto params = calc_launch_params(device_ptrs.size());
    sim_init_kernel<<<params.grid_size, params.block_size>>>(device_ptrs);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
