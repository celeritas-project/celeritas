//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PreStepAction.cu
//---------------------------------------------------------------------------//
#include "PreStepAction.hh"

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "PreStepExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch the pre-step action on device.
 */
void PreStepAction::step(CoreParams const& params, CoreStateDevice& state) const
{
    TrackExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), PreStepExecutor{}};

    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
