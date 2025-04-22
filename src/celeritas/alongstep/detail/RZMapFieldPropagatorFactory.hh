//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/RZMapFieldPropagatorFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/RZMapField.hh"  // IWYU pragma: associated
#include "celeritas/field/RZMapFieldData.hh"  // IWYU pragma: associated
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in an RZ map magnetic field.
 */
struct RZMapFieldPropagatorFactory
{
    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return make_mag_field_propagator<DormandPrinceIntegrator>(
            RZMapField{field},
            field.options,
            track.particle(),
            track.geometry());
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<RZMapFieldParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
