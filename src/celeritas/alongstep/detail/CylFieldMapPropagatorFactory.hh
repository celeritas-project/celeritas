//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/CylFieldMapPropagatorFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/CylFieldMap.hh"  // IWYU pragma: associated
#include "celeritas/field/CylFieldMapData.hh"  // IWYU pragma: associated
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in an Cyl map magnetic field.
 */
struct CylFieldMapPropagatorFactory
{
    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return make_mag_field_propagator<DormandPrinceStepper>(
            CylFieldMap{field},
            field.options,
            track.particle(),
            track.geometry());
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return true; }

    //// DATA ////

    NativeCRef<CylFieldMapParamsData> field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
