//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file XsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Quantity.hh"
#include "Interpolator.hh"
#include "UniformGrid.hh"
#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections on a uniform log grid.
 *
 * \todo Currently this is hard-coded to use "cross section grid data"
 * which have energy coordinates uniform in log space. This should
 * be expanded to handle multiple parameterizations of the energy grid (e.g.,
 * arbitrary spacing needed for the Livermore sampling) and of the value
 * interpolation (e.g. log interpolation). It might also make sense to get rid
 * of the "prime energy" and just use log-log interpolation instead, or do a
 * piecewise change in the interpolation instead of storing the cross section
 * scaled by the energy.
 *
 * \code
    XsCalculator calc_xs(xs_grid, xs_params.reals);
    real_type xs = calc_xs(particle);
   \endcode
 */
class XsCalculator
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
    XsCalculator(const XsGridData& grid, const Values& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the cross section at the given index
    inline CELER_FUNCTION real_type operator[](size_type index) const;

  private:
    const XsGridData& data_;
    const Values&     reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
XsCalculator::XsCalculator(const XsGridData& grid, const Values& values)
    : data_(grid), reals_(values)
{
    CELER_EXPECT(data_);
    CELER_ASSERT(grid.value.size() == data_.log_energy.size);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * If needed, we can add a "log(energy/MeV)" accessor if we constantly reuse
 * that value and don't want to repeat the `std::log` operation.
 */
CELER_FUNCTION real_type XsCalculator::operator()(Energy energy) const
{
    const UniformGrid loge_grid(data_.log_energy);
    const real_type   loge = std::log(energy.value());

    // Snap out-of-bounds values to closest grid points
    size_type lower_idx;
    real_type result;
    if (loge <= loge_grid.front())
    {
        lower_idx = 0;
        result    = this->get(lower_idx);
    }
    else if (loge >= loge_grid.back())
    {
        lower_idx = loge_grid.size() - 1;
        result    = this->get(lower_idx);
    }
    else
    {
        // Locate the energy bin
        lower_idx = loge_grid.find(loge);
        CELER_ASSERT(lower_idx + 1 < loge_grid.size());

        const real_type upper_energy = std::exp(loge_grid[lower_idx + 1]);
        real_type       upper_xs     = this->get(lower_idx + 1);
        if (lower_idx + 1 == data_.prime_index)
        {
            // Cross section data for the upper point has *already* been scaled
            // by E -- undo the scaling.
            upper_xs /= upper_energy;
        }

        // Interpolate *linearly* on energy using the lower_idx data.
        LinearInterpolator<real_type> interpolate_xs(
            {std::exp(loge_grid[lower_idx]), this->get(lower_idx)},
            {upper_energy, upper_xs});
        result = interpolate_xs(energy.value());
    }

    if (lower_idx >= data_.prime_index)
    {
        result /= energy.value();
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the cross section at the given index.
 */
CELER_FUNCTION real_type XsCalculator::operator[](size_type index) const
{
    const UniformGrid loge_grid(data_.log_energy);
    real_type         energy = std::exp(loge_grid[index]);
    real_type         result = this->get(index);

    if (index >= data_.prime_index)
    {
        result /= energy;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw cross section data at a particular index.
 */
CELER_FUNCTION real_type XsCalculator::get(size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
