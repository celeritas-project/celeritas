//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepUniformMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/em/params/FluctuationParams.hh"
#include "celeritas/em/params/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/ElossApplier.hh"
#include "detail/FieldFunctors.hh"
#include "detail/FieldTrackPropagator.hh"
#include "detail/FluctELoss.hh"
#include "detail/LinearTrackPropagator.hh"
#include "detail/MeanELoss.hh"
#include "detail/MscApplier.hh"
#include "detail/MscStepLimitApplier.hh"
#include "detail/PropagationApplier.hh"
#include "detail/TimeUpdater.hh"
#include "detail/TrackUpdater.hh"

// Field classes
#include "celeritas/field/UniformField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepUniformMscAction>
AlongStepUniformMscAction::from_params(ActionId id,
                                       CoreGeoParams const& geometry,
                                       MaterialParams const& materials,
                                       ParticleParams const& particles,
                                       Input const& field_input,
                                       SPConstMsc msc,
                                       bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepUniformMscAction>(
        id, geometry, field_input, std::move(fluct), msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with MSC data and field driver options.
 */
AlongStepUniformMscAction::AlongStepUniformMscAction(
    ActionId id,
    CoreGeoParams const& geometry,
    Input const& input,
    SPConstFluctuations fluct,
    SPConstMsc msc)
    : id_(id)
    , field_{std::make_shared<UniformFieldParams>(geometry, input)}
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(field_);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepUniformMscAction::~AlongStepUniformMscAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepUniformMscAction::step(CoreParams const& params,
                                     CoreStateHost& state) const
{
    using namespace ::celeritas::detail;
    auto execute_track = [&](CoreTrackView& track) {
        if (this->has_msc())
        {
            MscStepLimitApplier{UrbanMsc{msc_->ref<MemSpace::native>()}}(track);
        }
        if (IsInUniformField{field_->ref<MemSpace::native>()}(track))
        {
            PropagationApplier{FieldTrackPropagator<UniformField>{
                field_->ref<MemSpace::native>()}}(track);
        }
        else
        {
            PropagationApplier{LinearTrackPropagator{}}(track);
        }
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
    };

    return launch_action(
        *this,
        params,
        state,
        make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                       state.ptr(),
                                       this->action_id(),
                                       execute_track));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void AlongStepUniformMscAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
