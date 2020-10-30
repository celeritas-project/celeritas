//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagation.i.hh
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
CELER_FUNCTION LinearPropagator::LinearPropagator(GeoTrackView& track)
    : track_(track)
{
}

//---------------------------------------------------------------------------//
/*!
 * Move track by next_step(), which takes it to next volume boundary.
 */
CELER_FUNCTION
void LinearPropagator::operator()()
{
    this->apply_linear_step(track_.next_step() + track_.tolerance());

    // Update state
    track_.move_next_volume();
}

//---------------------------------------------------------------------------//
/*!
 * Move track by a user-provided distance.
 *
 * Step must be within current volume. User can ask next_step() for maximum
 * distance allowed before reaching volume boundary.
 *
 * \pre Assumes that next_step() has been properly called by client
 */
CELER_FUNCTION
void LinearPropagator::operator()(real_type dist)
{
    REQUIRE(dist > 0.);
    REQUIRE(dist <= track_.next_step());
    this->apply_linear_step(dist);

    if (std::fabs(track_.next_step()) < track_.tolerance())
    {
        track_.move_next_volume();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Do straight propagation to physics process or boundary and reduce next_step
 *
 * Scalar geometry length computation. The track is moved along track.dir()
 * direction by a distance track.next_step()
 */
CELER_FUNCTION
void LinearPropagator::apply_linear_step(real_type step)
{
    axpy(step, track_.dir(), &track_.pos());
    track_.next_step() -= step;
}

} // namespace celeritas
