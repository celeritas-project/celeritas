//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepCylMapFieldMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepCylMapFieldMscAction.hh"

#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/em/params/FluctuationParams.hh"  // IWYU pragma: keep
#include "celeritas/em/params/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/ElossApplier.hh"
#include "detail/FieldTrackPropagator.hh"
#include "detail/FluctELoss.hh"
#include "detail/MeanELoss.hh"
#include "detail/MscApplier.hh"
#include "detail/MscStepLimitApplier.hh"
#include "detail/PropagationApplier.hh"
#include "detail/TimeUpdater.hh"
#include "detail/TrackUpdater.hh"

// Field classes
#include "celeritas/field/CylMapField.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepCylMapFieldMscAction>
AlongStepCylMapFieldMscAction::from_params(ActionId id,
                                           MaterialParams const& materials,
                                           ParticleParams const& particles,
                                           CylMapFieldInput const& field_input,
                                           SPConstMsc const& msc,
                                           bool eloss_fluctuation)
{
    CELER_EXPECT(field_input);

    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepCylMapFieldMscAction>(
        id, field_input, std::move(fluct), msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID, energy loss parameters, and MSC.
 */
AlongStepCylMapFieldMscAction::AlongStepCylMapFieldMscAction(
    ActionId id,
    CylMapFieldInput const& input,
    SPConstFluctuations fluct,
    SPConstMsc msc)
    : id_(id)
    , field_{std::make_shared<CylMapFieldParams>(input)}
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(field_);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepCylMapFieldMscAction::step(CoreParams const& params,
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
        PropagationApplier{FieldTrackPropagator<CylMapField>{
            field_->ref<MemSpace::native>()}}(track);
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
#if !CELER_USE_DEVICE
void AlongStepCylMapFieldMscAction::step(CoreParams const&,
                                         CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
