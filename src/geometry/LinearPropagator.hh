//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeoTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate (move) a particle in a straight line.
 *
 * This can be called repeatedly until the track crosses a geometry boundary.
 */
class LinearPropagator
{
  public:
    //! Output results
    struct result_type
    {
        real_type distance; //!< Distance traveled
        bool      boundary; //!< True if hit a boundary before given distance
    };

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION LinearPropagator(GeoTrackView* track);

    // Move track to next volume boundary.
    inline CELER_FUNCTION result_type operator()();

    // Move track by a user-provided distance, or to the next boundary
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
    result_type result;
    result.distance = track_.move_to_boundary();
    result.boundary = true;
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

    if (dist >= track_.next_step())
    {
        // Move to boundary
        return (*this)();
    }

    result_type result;
    result.distance = track_.move_internal(dist);
    result.boundary = false;
    return resuult;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
