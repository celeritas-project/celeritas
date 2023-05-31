//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "AlongStep.hh"
#include "detail/FluctELoss.hh"
#include "detail/RZMapFieldTrackPropagator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepRZMapFieldMscAction>
AlongStepRZMapFieldMscAction::from_params(ActionId id,
                                          MaterialParams const& materials,
                                          ParticleParams const& particles,
                                          RZMapFieldInput const& field_input,
                                          SPConstMsc const& msc)
{
    SPConstFluctuations fluct
        = std::make_shared<FluctuationParams>(particles, materials);

    return std::make_shared<AlongStepRZMapFieldMscAction>(
        id, std::move(fluct), field_input, msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepRZMapFieldMscAction::AlongStepRZMapFieldMscAction(
    ActionId id,
    SPConstFluctuations fluct,
    RZMapFieldInput const& input,
    SPConstMsc msc)
    : id_(id), fluct_(std::move(fluct)), msc_(std::move(msc))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(input);
    CELER_EXPECT(fluct_);
    CELER_EXPECT(msc_);

    field_ = std::make_shared<RZMapFieldParams>(input);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepRZMapFieldMscAction::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    auto execute = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        AlongStep{
            UrbanMsc{msc_->ref<MemSpace::native>()},
            detail::RZMapFieldTrackPropagator{field_->ref<MemSpace::native>()},
            detail::FluctELoss{fluct_->ref<MemSpace::native>()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
