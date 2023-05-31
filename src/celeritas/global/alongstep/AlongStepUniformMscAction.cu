//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cu
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/AlongStepImpl.hh"
#include "detail/MeanELoss.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
__global__ void along_step_apply_msc_step_limit_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    DeviceCRef<UrbanMscData> const msc_data)
{
    auto execute = make_along_step_track_executor(
        *params,
        *state,
        along_step_id,
        detail::apply_msc_step_limit<UrbanMsc>,
        UrbanMsc{msc_data});
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_uniform_propagation_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    UniformFieldParams const field)
{
    auto execute = make_along_step_track_executor(
        *params,
        *state,
        along_step_id,
        detail::ApplyPropagation{},
        [&field](ParticleTrackView const& particle, GeoTrackView* geo) {
            return make_mag_field_propagator<DormandPrinceStepper>(
                UniformField(field.field), field.options, particle, geo);
        });
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_msc_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    DeviceCRef<UrbanMscData> const msc_data)
{
    auto execute = make_along_step_track_executor(*params,
                                                  *state,
                                                  along_step_id,
                                                  detail::apply_msc<UrbanMsc>,
                                                  UrbanMsc{msc_data});
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
__global__ void along_step_update_time_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id)
{
    auto execute = make_along_step_track_executor(
        *params, *state, along_step_id, detail::update_time);
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_mean_eloss_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id)
{
    using detail::MeanELoss;

    auto execute
        = make_along_step_track_executor(*params,
                                         *state,
                                         along_step_id,
                                         detail::apply_eloss<MeanELoss>,
                                         MeanELoss{});
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
__global__ void along_step_update_track_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id)
{
    auto execute = make_along_step_track_executor(
        *params, *state, along_step_id, detail::update_track);
    execute(KernelParamCalculator::thread_id());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
void AlongStepUniformMscAction::execute(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    CELER_LAUNCH_KERNEL(along_step_apply_msc_step_limit,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        device_data_.msc);
    CELER_LAUNCH_KERNEL(along_step_apply_uniform_propagation,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        field_params_);
    CELER_LAUNCH_KERNEL(along_step_apply_msc,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        device_data_.msc);
    CELER_LAUNCH_KERNEL(along_step_update_time,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id());
    CELER_LAUNCH_KERNEL(along_step_apply_mean_eloss,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id());
    CELER_LAUNCH_KERNEL(along_step_update_track,
                        celeritas::device().default_block_size(),
                        state.size(),
                        celeritas::device().stream(state.stream_id()).get(),
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
