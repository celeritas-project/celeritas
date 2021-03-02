//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include "base/ArrayUtils.hh"
#include "geometry/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
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
 * Move track by next_step(), which takes it to next volume boundary.
 */
CELER_FUNCTION
LinearPropagator::result_type LinearPropagator::operator()()
{
    result_type result;
    result.distance = track_.move_to_boundary();
    result.volume = track_.volume_id();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move track by a user-provided distance.
 *
 * Step must be positive. track_ will check for boundaries to be crossed.
 *
 * \pre Assumes that next_step() has been properly called by client
 */
CELER_FUNCTION
LinearPropagator::result_type LinearPropagator::operator()(real_type dist)
{
    CELER_EXPECT(dist > 0.);

    result_type result;
    result.distance = track_.move_by(dist);
    result.volume = track_.volume_id();
    return result;
}

} // namespace celeritas
