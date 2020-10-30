//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/NumericLimits.hh"
#include "GeoTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate (move) a particle in a straight line.
 */
class LinearPropagator
{
    using Initializer_t = GeoStateInitializer;

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION LinearPropagator(GeoTrackView& track);

    // Move track by next_step(), which takes it to next volume boundary.
    inline CELER_FUNCTION void operator()();

    // Move track by a user-provided distance.
    inline CELER_FUNCTION void operator()(real_type dist);

  private:
    // Fast move, update only the position.
    inline CELER_FUNCTION void apply_linear_step(real_type step);

  private:
    GeoTrackView& track_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "LinearPropagator.i.hh"
