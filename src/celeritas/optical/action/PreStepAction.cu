//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/PreStepAction.cu
//---------------------------------------------------------------------------//
#include "PreStepAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.device.hh"
#include "TrackSlotExecutor.hh"

#include "detail/PreStepExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the pre-step action on device.
 */
void PreStepAction::step(CoreParams const& params, CoreStateDevice& state) const
{
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), detail::PreStepExecutor{}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
