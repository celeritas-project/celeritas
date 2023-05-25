//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cu
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/InitTracksLauncher.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void init_tracks_kernel(detail::InitTracksLauncher launch)
{
    launch(KernelParamCalculator::thread_id());
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to initialize tracks.
 */
void InitializeTracksAction::launch_impl(CoreParams const& params,
                                         CoreStateDevice& state,
                                         size_type num_new_tracks) const
{
    CELER_LAUNCH_KERNEL(
        init_tracks,
        celeritas::device().default_block_size(),
        num_new_tracks,
        detail::InitTracksLauncher{params.ptr<MemSpace::device>(),
                                   state.ptr(),
                                   num_new_tracks,
                                   state.counters()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
