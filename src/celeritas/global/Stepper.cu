//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cu
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "CoreParams.hh"
#include "CoreState.hh"

#include "detail/SetGeneratedExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set the num_pending counter to the number of generated primaries.
 */
template<>
void Stepper<MemSpace::device>::set_generated()
{
    auto execute_thread
        = make_single_track_executor(params_->ptr<MemSpace::native>(),
                                     state_->ptr(),
                                     detail::SetGeneratedExecutor{});
    static KernelLauncher<decltype(execute_thread)> const launch_kernel(
        "set-generated");
    launch_kernel(1, state_->stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
