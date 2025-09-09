//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/AlongStepKernels.cu
//---------------------------------------------------------------------------//
#include "AlongStepKernels.hh"

#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "ElossApplier.hh"
#include "FluctELoss.hh"
#include "LinearPropagatorFactory.hh"
#include "MeanELoss.hh"
#include "MscApplier.hh"
#include "MscStepLimitApplier.hh"
#include "PropagationApplier.hh"
#include "TimeUpdater.hh"
#include "TrackUpdater.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Apply MSC step limiter (UrbanMsc)
void launch_limit_msc_step(CoreStepActionInterface const& action,
                           DeviceCRef<UrbanMscData> const& msc_data,
                           CoreParams const& params,
                           CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"limit-step-msc-urban"};
    auto execute_thread = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        action.action_id(),
        detail::MscStepLimitApplier{UrbanMsc{msc_data}});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "limit-step-msc-urban");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Apply linear propagation
void launch_propagate(CoreStepActionInterface const& action,
                      CoreParams const& params,
                      CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"propagate"};
    auto execute_thread = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        action.action_id(),
        detail::PropagationApplier{detail::LinearPropagatorFactory{}});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "propagate-linear");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Apply MSC scattering (UrbanMsc)
void launch_apply_msc(CoreStepActionInterface const& action,
                      DeviceCRef<UrbanMscData> const& msc_data,
                      CoreParams const& params,
                      CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"scatter-msc-urban"};
    auto execute_thread = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        action.action_id(),
        detail::MscApplier{UrbanMsc{msc_data}});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "scatter-msc-urban");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Update track times
void launch_update_time(CoreStepActionInterface const& action,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"update-time"};
    auto execute_thread
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         action.action_id(),
                                         detail::TimeUpdater{});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "update-time");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Apply energy loss with fluctuations
void launch_apply_eloss(CoreStepActionInterface const& action,
                        DeviceCRef<FluctuationData> const& fluct,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"apply-eloss-fluct"};
    auto execute_thread = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        action.action_id(),
        detail::ElossApplier{detail::FluctELoss{fluct}});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "apply-eloss-fluct");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Apply energy loss without fluctuations
void launch_apply_eloss(CoreStepActionInterface const& action,
                        CoreParams const& params,
                        CoreState<MemSpace::device>& state)
{
    ScopedProfiling profile_this{"apply-eloss"};
    auto execute_thread = make_along_step_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        action.action_id(),
        detail::ElossApplier{detail::MeanELoss{}});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "apply-eloss-mean");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
//! Update the track state at the end of along-step
void launch_update_track(CoreStepActionInterface const& action,
                         CoreParams const& params,
                         CoreState<MemSpace::device>& state)
{
    auto execute_thread
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         action.action_id(),
                                         detail::TrackUpdater{});
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(
        action, "update-track");
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
