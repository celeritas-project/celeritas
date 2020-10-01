//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagation.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/LinearPropagator.hh"
#include "base/ArrayUtils.hh"
#include "geometry/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//___________________________________________________________________________//
/*!
 * Do straight propagation to physics process or boundary and reduce next_step
 *
 * Scalar geometry length computation. The track is moved along track.dir()
 * direction by a distance track.next_step()
 */
CELER_FORCEINLINE_FUNCTION
void LinearPropagator::apply_linear_step(real_type step)
{
    axpy(step, track_.dir(), &track_.pos());
    track_.next_step() -= step;
}

//___________________________________________________________________________//
/*!
 * No step provided -> self-aware move to next boundary.
 *
 * State gets updated to next volume
 */
CELER_FORCEINLINE_FUNCTION
void LinearPropagator::operator()()
{
    this->apply_linear_step(track_.next_step() + track_.tolerance());

    // Update state
    track_.move_next_volume();
}

//___________________________________________________________________________//
/*!
 * User-provided step: check that move and update state if necessary.
 *
 * \pre Assumes that next_step() has been properly called by client
 */
CELER_FORCEINLINE_FUNCTION
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

} // namespace celeritas
