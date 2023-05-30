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
#include "corecel/sys/Stream.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveExecutor.hh"
#include "detail/ProcessSecondariesExecutor.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
__global__ void locate_alive_kernel(detail::LocateAliveExecutor launch)
{
    launch(KernelParamCalculator::thread_id());
}

__global__ void
process_secondaries_kernel(detail::ProcessSecondariesExecutor launch)
{
    launch(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to locate alive particles.
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
        celeritas::device().stream(core_state.stream_id()).get(),
        detail::LocateAliveExecutor{core_params.ptr<MemSpace::native>(),
                                    core_state.ptr()});
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to create track initializers from secondary particles.
 */
void ExtendFromSecondariesAction::process_secondaries(
    CoreParams const& core_params, CoreStateDevice& core_state) const
{
    CELER_LAUNCH_KERNEL(
        process_secondaries,
        celeritas::device().default_block_size(),
        core_state.size(),
        celeritas::device().stream(core_state.stream_id()).get(),
        detail::ProcessSecondariesExecutor{core_params.ptr<MemSpace::native>(),
                                           core_state.ptr(),
                                           core_state.counters()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
