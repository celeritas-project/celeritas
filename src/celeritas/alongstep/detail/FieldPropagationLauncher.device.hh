//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldPropagationLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>

#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "FieldFunctors.hh"
#include "FieldTrackPropagator.hh"
#include "PropagationApplier.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch the field propagation kernels for tracks satisfying a condition.
 *
 * If the safety sphere is enabled, tracks whose whole step fits inside it are
 * propagated first by a kernel that only integrates the field (\c
 * <label>-sphere ). The remaining tracks are then propagated by the kernel
 * that searches for boundaries ( \c <label> ). Each kernel is compiled with a
 * single propagator, so the sphere kernel doesn't carry the register and local
 * memory cost of the navigation code.
 */
template<class Field, class C>
void launch_field_propagation(CoreStepActionInterface const& action,
                              typename Field::ParamsRef const& field,
                              C const& condition,
                              std::string_view label,
                              CoreParams const& params,
                              CoreState<MemSpace::device>& state)
{
    auto const& options = field.options;
    if (options.use_safety_sphere)
    {
        auto execute_thread = ConditionalTrackExecutor{
            params.ptr<MemSpace::native>(),
            state.ptr(),
            IsStepInSafetySphere<C>{condition, options.delta_intersection},
            PropagationApplier{SafetyFieldTrackPropagator<Field>{field}}};
        static ActionLauncher<decltype(execute_thread)> const launch_kernel(
            action, std::string{label} + "-sphere");
        launch_kernel(action, params, state, execute_thread);
    }

    auto execute_thread = ConditionalTrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        IsStepOutsideSafetySphere<C>{
            condition, options.delta_intersection, options.use_safety_sphere},
        PropagationApplier{FieldTrackPropagator<Field>{field}}};
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(action,
                                                                        label);
    launch_kernel(action, params, state, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
