//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/em/msc/UrbanMsc.hh"  // IWYU pragma: associated
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "detail/AlongStepNeutral.hh"
#include "detail/FluctELoss.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepGeneralLinearAction>
AlongStepGeneralLinearAction::from_params(ActionId id,
                                          MaterialParams const& materials,
                                          ParticleParams const& particles,
                                          SPConstMsc const& msc,
                                          bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepGeneralLinearAction>(
        id, std::move(fluct), msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepGeneralLinearAction::AlongStepGeneralLinearAction(
    ActionId id, SPConstFluctuations fluct, SPConstMsc msc)
    : id_(id)
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
    , host_data_(fluct_, msc_)
    , device_data_(fluct_, msc_)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepGeneralLinearAction::~AlongStepGeneralLinearAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepGeneralLinearAction::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    auto execute = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        AlongStep{UrbanMsc{host_data_.msc},
                  detail::LinearTrackPropagator{},
                  detail::FluctELoss{host_data_.fluct}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepGeneralLinearAction::ExternalRefs<M>::ExternalRefs(
    SPConstFluctuations const& fluct_params, SPConstMsc const& msc_params)
{
    if (M == MemSpace::device && !celeritas::device())
    {
        // Skip device copy if disabled
        return;
    }

    if (fluct_params)
    {
        fluct = fluct_params->ref<M>();
    }
    if (msc_params)
    {
        msc = msc_params->ref<M>();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
