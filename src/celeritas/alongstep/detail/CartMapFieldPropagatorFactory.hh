//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/CartMapFieldPropagatorFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/CartMapField.hh"  // IWYU pragma: associated
#include "celeritas/field/CartMapFieldData.hh"  // IWYU pragma: associated
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a cartesian map magnetic field.
 */
struct CartMapFieldPropagatorFactory
{
    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return make_mag_field_propagator<DormandPrinceIntegrator>(
            CartMapField{field},
            field.options,
            track.particle(),
            track.geometry());
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<CartMapFieldParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
