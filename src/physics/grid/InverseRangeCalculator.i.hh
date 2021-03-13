//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InverseRangeCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Algorithms.hh"
#include "base/Assert.hh"
#include "Interpolator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from range data.
 *
 * The range is expected to be monotonically increaing with energy.
 * Lower-energy particles have shorter ranges.
 */
CELER_FUNCTION
InverseRangeCalculator::InverseRangeCalculator(const XsGridData& grid,
                                               const Values&     values)
    : log_energy_(grid.log_energy), range_(grid.value, values)
{
    CELER_EXPECT(range_.size() == log_energy_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy of a particle that has the given range.
 */
CELER_FUNCTION auto InverseRangeCalculator::operator()(real_type range) const
    -> Energy
{
    CELER_EXPECT(range >= 0 && range <= range_.back());

    if (range < range_.front())
    {
        // Very short range:  this corresponds to "energy < emin" for range
        // calculation: range = r[0] * sqrt(E / E[0])
        return Energy{std::exp(log_energy_.front())
                      * ipow<2>(range / range_.front())};
    }
    // Range should *never* exceed the longest range (highest energy) since
    // that should have limited the step
    if (CELER_UNLIKELY(range >= range_.back()))
    {
        CELER_ASSERT(range == range_.back());
        return Energy{std::exp(log_energy_.back())};
    }

    // Search for lower bin index
    auto idx = range_.find(range);
    CELER_ASSERT(idx + 1 < log_energy_.size());

    // Interpolate: 'x' = range, y = log energy
    LinearInterpolator<real_type> interpolate_log_energy(
        {range_[idx], std::exp(log_energy_[idx])},
        {range_[idx + 1], std::exp(log_energy_[idx + 1])});
    auto loge = interpolate_log_energy(range);
    return Energy{loge};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
