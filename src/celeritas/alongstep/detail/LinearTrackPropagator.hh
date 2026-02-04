//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/LinearTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/global/CoreTrackView.hh"

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
    CELER_FUNCTION Propagation operator()(CoreTrackView const& track) const
    {
        return LinearPropagator{track.geometry()}(track.sim().step_length());
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
