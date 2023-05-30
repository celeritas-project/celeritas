//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepNeutralAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepNeutralAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/LaunchAction.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepNeutral.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID.
 */
AlongStepNeutralAction::AlongStepNeutralAction(ActionId id) : id_(id)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepNeutralAction::execute(CoreParams const& params,
                                     CoreStateHost& state) const
{
    auto execute
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         this->action_id(),
                                         detail::AlongStepNeutralExecutor{});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void AlongStepNeutralAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
