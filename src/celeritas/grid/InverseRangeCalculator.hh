//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseRangeCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"

#include "Interpolator.hh"
#include "NonuniformGrid.hh"
#include "UniformGrid.hh"
#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the energy that would limit a particle to a particular range.
 *
 * This should provide the inverse of the result of \c RangeCalculator. The
 * given \c range is not allowed to be greater than the maximum range in the
 * physics data.
 *
 * The range must be monotonically increasing in energy, since it's defined as
 * the integral of the inverse of the stopping power (which is always
 * positive). For ranges shorter than the minimum energy in the table, the
 * resulting energy is scaled:
 * \f[
    E = E_\mathrm{min} \left( \frac{r}{r_\mathrm{min}} \right)^2
 * \f]
 * This scaling is the inverse of the off-the-end energy scaling in the
 * RangeCalculator.
 */
class InverseRangeCalculator
{
  public:
    //!@{
    //! Type aliases
    using Energy = Quantity<XsGridData::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    InverseRangeCalculator(XsGridData const& grid, Values const& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION Energy operator()(real_type range) const;

  private:
    UniformGrid log_energy_;
    NonuniformGrid<real_type> range_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from range data.
 *
 * The range is expected to be monotonically increaing with energy.
 * Lower-energy particles have shorter ranges.
 */
CELER_FUNCTION
InverseRangeCalculator::InverseRangeCalculator(XsGridData const& grid,
                                               Values const& values)
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
}  // namespace celeritas
