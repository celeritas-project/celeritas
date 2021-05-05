//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.i.hh
//---------------------------------------------------------------------------//

#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform y <- ax + y for OdeState.
 */
CELER_FUNCTION
void axpy(real_type a, const OdeState& x, OdeState* y)
{
    axpy(a, x.pos, &y->pos);
    axpy(a, x.mom, &y->mom);
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
    CELER_ASSERT(errpos2 >= 0.0);
    CELER_ASSERT(magvel2 > 0.0);

    errpos2 /= (eps_pos * eps_pos);
    errvel2 /= (magvel2 * eps_rel_max * eps_rel_max);

    // Return the square of the maximum truncation error
    return std::fmax(errpos2, errvel2);
}

//---------------------------------------------------------------------------//
/*!
 * Closest distance between the segmentfrom beg.pos (\em A) to end.pos(\em B)
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

    axpy(-1.0, beg_state.pos, &beg_mid);
    axpy(-1.0, beg_state.pos, &beg_end);

    Real3 cross = cross_product(beg_end, beg_mid);
    return std::sqrt(dot_product(cross, cross) / dot_product(beg_end, beg_end));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
