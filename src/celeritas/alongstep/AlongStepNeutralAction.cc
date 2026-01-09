//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepNeutralAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepNeutralAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepNeutralImpl.hh"  // IWYU pragma: associated

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
void AlongStepNeutralAction::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    using namespace ::celeritas::detail;

    return launch_action(
        *this,
        params,
        state,
        make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                       state.ptr(),
                                       this->action_id(),
                                       AlongStepNeutralExecutor{}));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void AlongStepNeutralAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
