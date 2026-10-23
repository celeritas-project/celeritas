//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldPropagationLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>

#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/user/FieldDiagnostic.hh"

#include "FieldTrackPropagator.hh"
#include "PropagationApplier.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Launch the field propagation kernel on device.
 *
 * \tparam Field Field type
 * \tparam C Condition that selects which tracks to propagate
 *
 * If the field diagnostic is present, this launches the instrumented kernel
 * (whose name has a \c -diag suffix) instead of the production kernel. Each
 * instantiation (one per field type and condition) has its own launchers.
 */
template<class Field, class C>
void launch_field_propagation(CoreStepActionInterface const& action,
                              std::string_view ext,
                              C&& is_applicable,
                              typename Field::ParamsRef const& field,
                              FieldDiagnostic const* diag,
                              CoreParams const& params,
                              CoreState<MemSpace::device>& state)
{
    if (!diag)
    {
        auto execute = ConditionalTrackExecutor{
            params.ptr<MemSpace::native>(),
            state.ptr(),
            celeritas::forward<C>(is_applicable),
            PropagationApplier{FieldTrackPropagator<Field>{field}}};
        static ActionLauncher<decltype(execute)> const launch_kernel(action,
                                                                     ext);
        launch_kernel(action, params, state, execute);
    }
    else
    {
        auto execute = ConditionalTrackExecutor{
            params.ptr<MemSpace::native>(),
            state.ptr(),
            celeritas::forward<C>(is_applicable),
            PropagationApplier{DiagnosticFieldTrackPropagator<Field>{
                field,
                diag->params<MemSpace::native>(),
                diag->state<MemSpace::native>(state.stream_id(),
                                              state.size())}}};
        static ActionLauncher<decltype(execute)> const launch_kernel(
            action, std::string(ext) + "-diag");
        launch_kernel(action, params, state, execute);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
