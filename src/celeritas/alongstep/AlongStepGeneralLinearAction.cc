//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepGeneralLinearAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/msc/UrbanMsc.hh"  // IWYU pragma: associated
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "detail/ElossApplier.hh"
#include "detail/FluctELoss.hh"  // IWYU pragma: associated
#include "detail/LinearTrackPropagator.hh"
#include "detail/MeanELoss.hh"  // IWYU pragma: associated
#include "detail/MscApplier.hh"
#include "detail/MscStepLimitApplier.hh"
#include "detail/PropagationApplier.hh"
#include "detail/TimeUpdater.hh"
#include "detail/TrackUpdater.hh"

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
    : id_(id), fluct_(std::move(fluct)), msc_(std::move(msc))
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
void AlongStepGeneralLinearAction::step(CoreParams const& params,
                                        CoreStateHost& state) const
{
    using namespace ::celeritas::detail;

    auto launch_impl = [&](auto&& execute_track) {
        return launch_action(
            *this,
            params,
            state,
            make_along_step_track_executor(
                params.ptr<MemSpace::native>(),
                state.ptr(),
                this->action_id(),
                std::forward<decltype(execute_track)>(execute_track)));
    };

    launch_impl([&](CoreTrackView& track) {
        if (this->has_msc())
        {
            MscStepLimitApplier{UrbanMsc{msc_->ref<MemSpace::native>()}}(track);
        }
        PropagationApplier{LinearTrackPropagator{}}(track);
        if (this->has_msc())
        {
            MscApplier{UrbanMsc{msc_->ref<MemSpace::native>()}}(track);
        }
        TimeUpdater{}(track);
        if (this->has_fluct())
        {
            ElossApplier{FluctELoss{fluct_->ref<MemSpace::native>()}}(track);
        }
        else
        {
            ElossApplier{MeanELoss{}}(track);
        }
        TrackUpdater{}(track);
    });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
