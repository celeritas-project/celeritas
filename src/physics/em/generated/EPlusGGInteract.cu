//----------------------------------*-cu-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteract.cu
//! \note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "../detail/EPlusGGLauncher.hh"

using namespace celeritas::detail;

namespace celeritas
{
namespace generated
{
namespace
{
__global__ void eplusgg_interact_kernel(
    const detail::EPlusGGDeviceRef eplusgg_data,
    const ModelInteractRef<MemSpace::device> model)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    detail::EPlusGGLauncher<MemSpace::device> launch(eplusgg_data, model);
    launch(tid);
}
} // namespace

void eplusgg_interact(
    const detail::EPlusGGDeviceRef& eplusgg_data,
    const ModelInteractRef<MemSpace::device>& model)
{
    CELER_EXPECT(eplusgg_data);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        eplusgg_interact_kernel, "eplusgg_interact");
    auto params = calc_kernel_params(model.states.size());
    eplusgg_interact_kernel<<<params.grid_size, params.block_size>>>(
        eplusgg_data, model);
    CELER_CUDA_CHECK_ERROR();
}

} // namespace generated
} // namespace celeritas
