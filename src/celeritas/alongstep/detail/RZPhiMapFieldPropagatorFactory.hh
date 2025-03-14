//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/RZPhiMapFieldPropagatorFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/RZPhiMapField.hh"  // IWYU pragma: associated
#include "celeritas/field/RZPhiMapFieldData.hh"  // IWYU pragma: associated
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in an RZ map magnetic field.
 */
struct RZPhiMapFieldPropagatorFactory
{
    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return make_mag_field_propagator<DormandPrinceStepper>(
            RZPhiMapField{field},
            field.options,
            track.particle(),
            track.geometry());
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<RZPhiMapFieldParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
