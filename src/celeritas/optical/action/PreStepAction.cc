//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/PreStepAction.cc
//---------------------------------------------------------------------------//
#include "PreStepAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/PreStepExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
PreStepAction::PreStepAction(ActionId aid)
    : ConcreteAction(aid, "pre-step", "update beginning-of-step state")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the pre-step action on host.
 */
void PreStepAction::step(CoreParams const& params, CoreStateHost& state) const
{
    TrackSlotExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), detail::PreStepExecutor{}};
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void PreStepAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
