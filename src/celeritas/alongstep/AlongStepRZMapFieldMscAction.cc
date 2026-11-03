//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepRZMapFieldMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/em/params/FluctuationParams.hh"  // IWYU pragma: keep
#include "celeritas/em/params/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "detail/ElossApplier.hh"
#include "detail/FieldPropagationLauncher.hh"
#include "detail/FluctELoss.hh"
#include "detail/MeanELoss.hh"
#include "detail/MscApplier.hh"
#include "detail/MscStepLimitApplier.hh"
#include "detail/TimeUpdater.hh"
#include "detail/TrackUpdater.hh"

// Field classes
#include "celeritas/field/RZMapField.hh"

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
                                          SPConstMsc const& msc,
                                          bool eloss_fluctuation,
                                          SPConstFieldDiagnostic field_diag)
{
    CELER_EXPECT(field_input);

    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepRZMapFieldMscAction>(
        id, field_input, std::move(fluct), msc, std::move(field_diag));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID, energy loss parameters, and MSC.
 *
 * If a field diagnostic is given, the propagation kernel is instrumented.
 */
AlongStepRZMapFieldMscAction::AlongStepRZMapFieldMscAction(
    ActionId id,
    RZMapFieldInput const& input,
    SPConstFluctuations fluct,
    SPConstMsc msc,
    SPConstFieldDiagnostic field_diag)
    : id_(id)
    , field_{std::make_shared<RZMapFieldParams>(input)}
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
    , field_diag_(std::move(field_diag))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(field_);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepRZMapFieldMscAction::step(CoreParams const& params,
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

    auto propagate = make_host_field_propagation<RZMapField>(
        field_->ref<MemSpace::native>(), field_diag_.get(), state);

    launch_impl([&](CoreTrackView& track) {
        if (this->has_msc())
        {
            MscStepLimitApplier{UrbanMsc{msc_->ref<MemSpace::native>()}}(track);
        }
        propagate(track);
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
void AlongStepRZMapFieldMscAction::step(CoreParams const&,
                                        CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
