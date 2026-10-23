//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldPropagationLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/user/FieldDiagnostic.hh"

#include "FieldTrackPropagator.hh"
#include "PropagationApplier.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a host function that propagates a track in a field.
 *
 * The host along-step loops apply all along-step operations to each track in
 * a single function, so the choice between the production and instrumented
 * propagation is made for each track. The device code instead launches one of
 * two kernels: see \c launch_field_propagation .
 */
template<class Field>
auto make_host_field_propagation(typename Field::ParamsRef const& field,
                                 FieldDiagnostic const* diag,
                                 CoreState<MemSpace::host>& state)
{
    using DiagParamsRef = HostCRef<FieldDiagnosticParamsData>;
    using DiagStateRef = HostRef<FieldDiagnosticStateData>;

    DiagParamsRef const* diag_params = nullptr;
    DiagStateRef const* diag_state = nullptr;
    if (diag)
    {
        diag_params = &diag->params<MemSpace::host>();
        diag_state
            = &diag->state<MemSpace::host>(state.stream_id(), state.size());
    }

    return [field, diag_params, diag_state](CoreTrackView& track) {
        if (diag_state)
        {
            PropagationApplier{DiagnosticFieldTrackPropagator<Field>{
                field, *diag_params, *diag_state}}(track);
        }
        else
        {
            PropagationApplier{FieldTrackPropagator<Field>{field}}(track);
        }
    };
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
