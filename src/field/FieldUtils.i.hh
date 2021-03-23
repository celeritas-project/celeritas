//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
// Perform y <- ax + y for OdeState
CELER_FUNCTION
void axpy(real_type a, const OdeState& x, OdeState* y)
{
    axpy(a, x.pos, &y->pos);
    axpy(a, x.mom, &y->mom);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the stepper truncation error: max(pos_error^2, scale*mom_error^2)
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
    CELER_ASSERT(errpos2 > 0.0);
    CELER_ASSERT(magvel2 > 0.0);

    errpos2 /= (eps_pos * eps_pos);
    errvel2 /= (magvel2 * eps_rel_max * eps_rel_max);

    // Return the square of the maximum truncation error
    return std::fmax(errpos2, errvel2);
}

//---------------------------------------------------------------------------//
/*!
 * Closest distance between the chord (from beg_state.pos to end_state.pos)
 * and the mid-state
 */
CELER_FUNCTION real_type distance_chord(const OdeState& beg_state,
                                        const OdeState& mid_state,
                                        const OdeState& end_state)
{
    Real3 beg_side{0, 0, 0};
    Real3 end_side{0, 0, 0};
    axpy(-1.0, beg_state.pos, &beg_side);
    axpy(-1.0, end_state.pos, &end_side);

    real_type beg_dist2 = dot_product(mid_state.pos, beg_side);
    real_type end_dist2 = dot_product(mid_state.pos, end_side);

    return std::sqrt(beg_dist2 * end_dist2 / (beg_dist2 + end_dist2));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
