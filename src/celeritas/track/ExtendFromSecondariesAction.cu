//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveLauncher.hh"
#include "detail/ProcessSecondariesLauncher.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void locate_alive_kernel(detail::LocateAliveLauncher launch)
{
    launch(KernelParamCalculator::thread_id());
}

__global__ void
process_secondaries_kernel(detail::ProcessSecondariesLauncher launch)
{
    launch(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to locate alive particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::locate_alive(CoreParams const& core_params,
                                               CoreStateDevice& core_state) const
{
    CELER_LAUNCH_KERNEL(
        locate_alive,
        celeritas::device().default_block_size(),
        core_state.size(),
        detail::LocateAliveLauncher{core_params.ptr<MemSpace::native>(),
                                    core_state.ptr()});
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to process secondary particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::process_secondaries(
    CoreParams const& core_params, CoreStateDevice& core_state) const
{
    CELER_LAUNCH_KERNEL(
        process_secondaries,
        celeritas::device().default_block_size(),
        core_state.size(),
        detail::ProcessSecondariesLauncher{core_params.ptr<MemSpace::native>(),
                                           core_state.ptr(),
                                           core_state.ref().init.scalars});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
