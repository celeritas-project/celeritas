//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Algorithms.hh"
#include "base/Array.hh"
#include "base/ArrayUtils.hh"
#include "base/Assert.hh"
#include "field/Types.hh"
#include "sim/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Return y <- ax for a real variable
template<class T>
inline CELER_FUNCTION Array<T, 3> ax(T a, const Array<T, 3>& x);

//---------------------------------------------------------------------------//
// Calculate the direction between the source and destination
struct Chord
{
    real_type length;
    Real3     dir;
};
inline CELER_FUNCTION Chord make_chord(const Real3& src, const Real3& dst);

//---------------------------------------------------------------------------//
inline CELER_FUNCTION bool is_intercept_close(const Real3& pos,
                                              const Real3& dir,
                                              real_type    distance,
                                              const Real3& target,
                                              real_type    tolerance);

//---------------------------------------------------------------------------//
// Evaluate the stepper truncation error
inline CELER_FUNCTION real_type truncation_error(real_type       step,
                                                 real_type       eps_rel_max,
                                                 const OdeState& beg_state,
                                                 const OdeState& err_state);

// Closest distance between the chord and the mid-state
inline CELER_FUNCTION real_type distance_chord(const OdeState& beg_state,
                                               const OdeState& mid_state,
                                               const OdeState& end_state);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return y <- ax for a real variable
 */
template<class T>
CELER_FUNCTION Array<T, 3> ax(T a, const Array<T, 3>& x)
{
    Array<T, 3> result;
    for (size_type i = 0; i != 3; ++i)
    {
        result[i] = a * x[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the direction between the source and destination.
 */
CELER_FUNCTION Chord make_chord(const Real3& src, const Real3& dst)
{
    Chord result;
    for (size_type i = 0; i != 3; ++i)
    {
        result.dir[i] = dst[i] - src[i];
    }
    result.length        = norm(result.dir);
    const real_type norm = 1 / result.length;
    for (size_type i = 0; i != 3; ++i)
    {
        result.dir[i] *= norm;
    }
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
CELER_FUNCTION bool is_intercept_close(const Real3& pos,
                                       const Real3& dir,
                                       real_type    distance,
                                       const Real3& target,
                                       real_type    tolerance)
{
    real_type delta_sq = 0;
    for (size_type i = 0; i != 3; ++i)
    {
        delta_sq += ipow<2>(pos[i] - target[i] + distance * dir[i]);
    }
    return delta_sq <= ipow<2>(tolerance);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the stepper truncation error square:
 * \f$ \Delta = max (\delta_{pos}^{2}, \epsilon \delta_{mom}^{2}) \f$
 */
CELER_FUNCTION real_type truncation_error(real_type       step,
                                          real_type       eps_rel_max,
                                          const OdeState& beg_state,
                                          const OdeState& err_state)
{
    // Evaluate tolerance and squre of the position and momentum accuracy
    real_type eps_pos = eps_rel_max * step;

    real_type magvel2 = dot_product(beg_state.mom, beg_state.mom);
    real_type errpos2 = dot_product(err_state.pos, err_state.pos);
    real_type errvel2 = dot_product(err_state.mom, err_state.mom);

    // Scale relative to a required tolerance
    CELER_ASSERT(errpos2 >= 0);
    CELER_ASSERT(magvel2 > 0);

    errpos2 /= ipow<2>(eps_pos);
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
CELER_FUNCTION real_type distance_chord(const OdeState& beg_state,
                                        const OdeState& mid_state,
                                        const OdeState& end_state)
{
    Real3 beg_mid = mid_state.pos;
    Real3 beg_end = end_state.pos;

    for (size_type i = 0; i != 3; ++i)
    {
        beg_mid[i] = mid_state.pos[i] - beg_state.pos[i];
        beg_end[i] = end_state.pos[i] - beg_state.pos[i];
    }

    Real3 cross = cross_product(beg_end, beg_mid);
    return std::sqrt(dot_product(cross, cross) / dot_product(beg_end, beg_end));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
