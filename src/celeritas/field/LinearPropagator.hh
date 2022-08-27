//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"
#include "celeritas/geo/GeoTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate (move) a particle in a straight line.
 */
class LinearPropagator
{
  public:
    //!@{
    //! Type aliases
    using result_type = Propagation;
    //!@}

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION LinearPropagator(GeoTrackView* track);

    // Move track to next volume boundary.
    inline CELER_FUNCTION result_type operator()();

    // Move track up to a user-provided distance, up to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

  private:
    GeoTrackView& track_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION LinearPropagator::LinearPropagator(GeoTrackView* track)
    : track_(*track)
{
    CELER_EXPECT(track);
}

//---------------------------------------------------------------------------//
/*!
 * Move track to next volume boundary.
 */
CELER_FUNCTION
LinearPropagator::result_type LinearPropagator::operator()()
{
    CELER_EXPECT(!track_.is_outside());

    result_type result = track_.find_next_step();
    CELER_ASSERT(result.boundary);
    track_.move_to_boundary();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move track by a user-provided distance up to the next boundary.
 */
CELER_FUNCTION
LinearPropagator::result_type LinearPropagator::operator()(real_type dist)
{
    CELER_EXPECT(dist > 0);

    result_type result = track_.find_next_step(dist);

    if (result.boundary)
    {
        track_.move_to_boundary();
    }
    else
    {
        CELER_ASSERT(dist == result.distance);
        track_.move_internal(dist);
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
