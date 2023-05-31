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
#include "celeritas/em/UrbanMscParams.hh"  // IWYU pragma: keep
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/DormandPrinceStepper.hh"  // IWYU pragma: associated
#include "celeritas/field/FieldDriverOptions.hh"  // IWYU pragma: associated
#include "celeritas/field/MakeMagFieldPropagator.hh"  // IWYU pragma: associated
#include "celeritas/field/UniformField.hh"  // IWYU pragma: associated
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "AlongStep.hh"
#include "detail/MeanELoss.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with MSC data and field driver options.
 */
AlongStepUniformMscAction::AlongStepUniformMscAction(
    ActionId id, UniformFieldParams const& field_params, SPConstMsc msc)
    : id_(id)
    , msc_(std::move(msc))
    , field_params_(field_params)
    , host_data_(msc_)
    , device_data_(msc_)
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
    auto execute = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        AlongStep{
            UrbanMsc{host_data_.msc},
            [field = field_params_](ParticleTrackView const& particle,
                                    GeoTrackView* geo) {
                return make_mag_field_propagator<DormandPrinceStepper>(
                    UniformField(field.field), field.options, particle, geo);
            },
            detail::MeanELoss{}});
    return launch_action(*this, params, state, execute);
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
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepUniformMscAction::ExternalRefs<M>::ExternalRefs(
    SPConstMsc const& msc_params)
{
    if (M == MemSpace::device && !celeritas::device())
    {
        // Skip device copy if disabled
        return;
    }

    if (msc_params)
    {
        msc = get_ref<M>(*msc_params);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
