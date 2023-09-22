//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "AlongStep.hh"
#include "detail/FluctELoss.hh"
#include "detail/MeanELoss.hh"
#include "detail/UniformFieldPropagatorFactory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepUniformMscAction>
AlongStepUniformMscAction::from_params(ActionId id,
                                       MaterialParams const& materials,
                                       ParticleParams const& particles,
                                       UniformFieldParams const& field_params,
                                       SPConstMsc msc,
                                       bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepUniformMscAction>(
        id, field_params, std::move(fluct), msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with MSC data and field driver options.
 */
AlongStepUniformMscAction::AlongStepUniformMscAction(
    ActionId id,
    UniformFieldParams const& field_params,
    SPConstFluctuations fluct,
    SPConstMsc msc)
    : id_(id)
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
    , field_params_(field_params)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepUniformMscAction::~AlongStepUniformMscAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepUniformMscAction::execute(CoreParams const& params,
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

    if (msc_)
    {
        launch_impl(
            MscStepLimitApplier{UrbanMsc{msc_->ref<MemSpace::native>()}});
    }
    launch_impl(
        PropagationApplier{UniformFieldPropagatorFactory{field_params_}});
    if (msc_)
    {
        launch_impl(MscApplier{UrbanMsc{msc_->ref<MemSpace::native>()}});
    }
    launch_impl(detail::TimeUpdater{});
    if (fluct_)
    {
        launch_impl(ElossApplier{FluctELoss{fluct_->ref<MemSpace::native>()}});
    }
    else
    {
        launch_impl(ElossApplier{MeanELoss{}});
    }
    launch_impl(TrackUpdater{});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void AlongStepUniformMscAction::execute(CoreParams const&,
                                        CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
