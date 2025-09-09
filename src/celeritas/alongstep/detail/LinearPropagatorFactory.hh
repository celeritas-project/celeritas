//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/LinearPropagatorFactory.hh
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
struct LinearPropagatorFactory
{
    CELER_FUNCTION decltype(auto) operator()(CoreTrackView const& track) const
    {
        return LinearPropagator{track.geometry()};
    }

    static CELER_CONSTEXPR_FUNCTION bool tracks_can_loop() { return false; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
