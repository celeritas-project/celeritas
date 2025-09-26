//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/AlongStepAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.device.hh"
#include "TrackSlotExecutor.hh"

#include "detail/AlongStepExecutor.hh"
#include "detail/PropagateExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepAction::step(CoreParams const& params,
                           CoreStateDevice& state) const
{
    {
        // Propagate
        auto execute = make_active_volumetric_thread_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            detail::PropagateExecutor{});

        static ActionLauncher<decltype(execute)> const launch_kernel(
            *this, "propagate");
        launch_kernel(state, execute);
    }
    {
        // Update state
        auto execute = make_active_volumetric_thread_executor(
            params.ptr<MemSpace::native>(),
            state.ptr(),
            detail::AlongStepExecutor{});

        static ActionLauncher<decltype(execute)> const launch_kernel(*this);
        launch_kernel(state, execute);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
