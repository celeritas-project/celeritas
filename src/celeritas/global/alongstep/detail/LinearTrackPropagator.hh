//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/LinearTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/geo/GeoTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a propagator for neutral particles or no fields.
 */
struct LinearTrackPropagator
{
    CELER_FUNCTION Propagation operator()(CoreTrackView const& track,
                                          real_type max_step) const
    {
        return LinearPropagator{track.make_geo_view()}(max_step);
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return false; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
