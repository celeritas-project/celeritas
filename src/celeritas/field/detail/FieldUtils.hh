//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "corecel/math/ArrayOperators.hh"
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
    result.dir = dst - src;
    result.length = norm(result.dir);
    result.dir /= result.length;
    return result;
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
    real_type delta_sq = 0;
    for (int i = 0; i < 3; ++i)
    {
        delta_sq += ipow<2>(pos[i] - target[i] + distance * dir[i]);
    }
    return delta_sq <= ipow<2>(tolerance);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the square of the relative stepper truncation error.
 *
 * \f$ \max(\delta_\textrm{pos}^{2}, \epsilon \delta_\textrm{mom}^{2}) \f$
 *
 * The return value is the square of \c dyerr in
 * \c G4MagIntegratorDriver::AccurateAdvance .
 */
inline CELER_FUNCTION real_type rel_err_sq(OdeState const& err_state,
                                           real_type step,
                                           Real3 const& mom)
{
    CELER_EXPECT(step > 0);

    // Evaluate square of the position and momentum accuracy
    real_type errpos2 = dot_product(err_state.pos, err_state.pos);
    real_type errvel2 = dot_product(err_state.mom, err_state.mom);

    // Scale position error relative to step
    errpos2 /= ipow<2>(step);
    // Scale momentum error relative to starting momentum
    errvel2 /= dot_product(mom, mom);

    real_type result = max(errpos2, errvel2);
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Closest distance between the segment from beg.pos (\em A) to end.pos(\em B)
 * and the mid.pos (\em M):
 * \f[
 *   d = |\vec{AM}| \sin(\theta) = \frac{\vec{AM} \times \vec{AB}}{|\vec{AB}|}
 * \f]
 */
inline CELER_FUNCTION real_type distance_chord(Real3 const& beg_pos,
                                               Real3 const& mid_pos,
                                               Real3 const& end_pos)
{
    Real3 beg_mid;
    Real3 beg_end;

    for (int i = 0; i < 3; ++i)
    {
        beg_mid[i] = mid_pos[i] - beg_pos[i];
        beg_end[i] = end_pos[i] - beg_pos[i];
    }

    Real3 cross = cross_product(beg_end, beg_mid);
    return std::sqrt(dot_product(cross, cross) / dot_product(beg_end, beg_end));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
