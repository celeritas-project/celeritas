//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/RZMapFieldTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/RZMapField.hh"  // IWYU pragma: associated
#include "celeritas/field/RZMapFieldData.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in an RZ map magnetic field.
 */
struct RZMapFieldTrackPropagator
{
    CELER_FUNCTION Propagation operator()(CoreTrackView const& track,
                                          real_type max_step) const
    {
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            RZMapField{field},
            field.options,
            track.make_particle_view(),
            track.make_geo_view());
        return propagate(max_step);
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<RZMapFieldParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
