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
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/track/TrackInitParams.hh"

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
    ThreadId const offset,
    DeviceCRef<UrbanMscData> const msc_data)
{
    auto launch = make_along_step_track_launcher(
        *params,
        *state,
        along_step_id,
        detail::apply_msc_step_limit<UrbanMsc>,
        UrbanMsc{msc_data});
    launch(KernelParamCalculator::thread_id() + offset.get());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_uniform_propagation_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    ThreadId const offset,
    UniformFieldParams const field)
{
    auto launch = make_along_step_track_launcher(
        *params,
        *state,
        along_step_id,
        detail::ApplyPropagation{},
        [&field](ParticleTrackView const& particle, GeoTrackView* geo) {
            return make_mag_field_propagator<DormandPrinceStepper>(
                UniformField(field.field), field.options, particle, geo);
        });
    launch(KernelParamCalculator::thread_id() + offset.get());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_msc_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    ThreadId const offset,
    DeviceCRef<UrbanMscData> const msc_data)
{
    auto launch = make_along_step_track_launcher(*params,
                                                 *state,
                                                 along_step_id,
                                                 detail::apply_msc<UrbanMsc>,
                                                 UrbanMsc{msc_data});
    launch(KernelParamCalculator::thread_id() + offset.get());
}

//---------------------------------------------------------------------------//
__global__ void along_step_update_time_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    ThreadId const offset)
{
    auto launch = make_along_step_track_launcher(
        *params, *state, along_step_id, detail::update_time);
    launch(KernelParamCalculator::thread_id() + offset.get());
}

//---------------------------------------------------------------------------//
__global__ void along_step_apply_mean_eloss_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    ThreadId const offset)
{
    using detail::MeanELoss;

    auto launch = make_along_step_track_launcher(*params,
                                                 *state,
                                                 along_step_id,
                                                 detail::apply_eloss<MeanELoss>,
                                                 MeanELoss{});
    launch(KernelParamCalculator::thread_id() + offset.get());
}

//---------------------------------------------------------------------------//
__global__ void along_step_update_track_kernel(
    CRefPtr<CoreParamsData, MemSpace::device> const params,
    RefPtr<CoreStateData, MemSpace::device> const state,
    ActionId const along_step_id,
    ThreadId const offset)
{
    auto launch = make_along_step_track_launcher(
        *params, *state, along_step_id, detail::update_track);
    launch(KernelParamCalculator::thread_id() + offset.get());
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
    // TODO: function to round up, single check with tuple
    auto const grid_size = [&] {
        if (params.init()->host_ref().track_order
            == TrackOrder::sort_along_step_action)
        {
            auto action_range = state.get_action_range(this->action_id());
            auto n_threads = action_range.size();
            auto bs = celeritas::device().default_block_size();
            return n_threads + bs - (n_threads % bs);
        }
        return state.size();
    }();
    auto const offset = [&] {
        if (params.init()->host_ref().track_order
            == TrackOrder::sort_along_step_action)
        {
            auto action_range = state.get_action_range(this->action_id());
            return action_range.front();
        }
        return ThreadId{0};
    }();
    CELER_LAUNCH_KERNEL(along_step_apply_msc_step_limit,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset,
                        device_data_.msc);
    CELER_LAUNCH_KERNEL(along_step_apply_uniform_propagation,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset,
                        field_params_);
    CELER_LAUNCH_KERNEL(along_step_apply_msc,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset,
                        device_data_.msc);
    CELER_LAUNCH_KERNEL(along_step_update_time,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset);
    CELER_LAUNCH_KERNEL(along_step_apply_mean_eloss,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset);
    CELER_LAUNCH_KERNEL(along_step_update_track,
                        celeritas::device().default_block_size(),
                        grid_size,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        this->action_id(),
                        offset);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
