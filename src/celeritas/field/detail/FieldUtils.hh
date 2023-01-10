//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/FieldUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Types.hh"

#include "../Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HELPER STRUCTS
//---------------------------------------------------------------------------//

//! Calculated line segment between two points
struct Chord
{
    real_type length;
    Real3 dir;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return y <- ax for a real variable
 */
template<class T>
inline CELER_FUNCTION Array<T, 3> ax(T a, Array<T, 3> const& x)
{
    Array<T, 3> result;
    for (int i = 0; i < 3; ++i)
    {
        result[i] = a * x[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the direction between the source and destination.
 */
inline CELER_FUNCTION Chord make_chord(Real3 const& src, Real3 const& dst)
{
    Chord result;
    for (int i = 0; i < 3; ++i)
    {
        result.dir[i] = dst[i] - src[i];
    }
    result.length = norm(result.dir);
    const real_type norm = 1 / result.length;
    for (int i = 0; i < 3; ++i)
    {
        result.dir[i] *= norm;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the distance between a target point and a line segment's endpoint.
 *
 * This is equivalent to:
 * \code
     Real3 temp = pos;
     axpy(distance, dir, &pos);

     return ipow<2>(distance(pos, target));
 * \endcode
 */
inline CELER_FUNCTION real_type calc_miss_distance(Real3 const& pos,
                                                   Real3 const& dir,
                                                   real_type distance,
                                                   Real3 const& target)
{
    real_type delta_sq = 0;
    for (int i = 0; i < 3; ++i)
    {
        delta_sq += ipow<2>(pos[i] - target[i] + distance * dir[i]);
    }
    return delta_sq;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the straight-line position is within a distance of the target.
 *
 * This is equivalent to:
 * \code
     Real3 temp = pos;
     axpy(distance, dir, &pos);

     return distance(pos, target) <= tolerance;
 * \endcode
 */
inline CELER_FUNCTION bool is_intercept_close(Real3 const& pos,
                                              Real3 const& dir,
                                              real_type distance,
                                              Real3 const& target,
                                              real_type tolerance)
{
    return calc_miss_distance(pos, dir, distance, target) <= ipow<2>(tolerance);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the stepper truncation error square:
 * \f$ \Delta = max (\delta_{pos}^{2}, \epsilon \delta_{mom}^{2}) \f$
 */
inline CELER_FUNCTION real_type truncation_error(real_type step,
                                                 real_type eps_rel_max,
                                                 OdeState const& beg_state,
                                                 OdeState const& err_state)
{
    CELER_EXPECT(step > 0);
    CELER_EXPECT(eps_rel_max > 0);

    // Evaluate tolerance and squre of the position and momentum accuracy

    real_type magvel2 = dot_product(beg_state.mom, beg_state.mom);
    real_type errpos2 = dot_product(err_state.pos, err_state.pos);
    real_type errvel2 = dot_product(err_state.mom, err_state.mom);

    // Scale relative to a required tolerance
    CELER_ASSERT(errpos2 >= 0);
    CELER_ASSERT(magvel2 > 0);

    errpos2 /= ipow<2>(eps_rel_max * step);
    errvel2 /= (magvel2 * ipow<2>(eps_rel_max));

    // Return the square of the maximum truncation error
    return max(errpos2, errvel2);
}

//---------------------------------------------------------------------------//
/*!
 * Closest distance between the segment from beg.pos (\em A) to end.pos(\em B)
 * and the mid.pos (\em M):
 * \f[
 *   d = |\vec{AM}| \sin(\theta) = \frac{\vec{AM} \times \vec{AB}}{|\vec{AB}|}
 * \f]
 */
inline CELER_FUNCTION real_type distance_chord(OdeState const& beg_state,
                                               OdeState const& mid_state,
                                               OdeState const& end_state)
{
    Real3 beg_mid;
    Real3 beg_end;

    for (int i = 0; i < 3; ++i)
    {
        beg_mid[i] = mid_state.pos[i] - beg_state.pos[i];
        beg_end[i] = end_state.pos[i] - beg_state.pos[i];
    }

    Real3 cross = cross_product(beg_end, beg_mid);
    return std::sqrt(dot_product(cross, cross) / dot_product(beg_end, beg_end));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
