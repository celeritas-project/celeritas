//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/PreStepAction.cc
//---------------------------------------------------------------------------//
#include "PreStepAction.hh"

#include <utility>

#include "corecel/Types.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "PreStepExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an action ID.
 */
PreStepAction::PreStepAction(ActionId aid)
    : StaticConcreteAction(aid, "pre-step", "update beginning-of-step state")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the pre-step action on host.
 */
void PreStepAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), PreStepExecutor{}};
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
