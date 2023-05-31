//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/UniformFieldTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/DormandPrinceStepper.hh"  // IWYU pragma: associated
#include "celeritas/field/MakeMagFieldPropagator.hh"  // IWYU pragma: associated
#include "celeritas/field/UniformField.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a uniform magnetic field.
 */
struct UniformFieldTrackPropagator
{
    CELER_FUNCTION Propagation operator()(CoreTrackView const& track,
                                          real_type max_step) const
    {
        auto geo = track.make_geo_view();
        auto particle = track.make_particle_view();
        auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
            UniformField(field.field), field.options, particle, &geo);
        return propagate(max_step);
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    UniformFieldParams field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
