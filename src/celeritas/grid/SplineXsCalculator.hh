//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Quantity.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections on a uniform log grid with an input
 * spline-order.
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
    SplineXsCalculator calc_xs(xs_grid, xs_params.reals, order);
    real_type xs = calc_xs(particle);
   \endcode
 */
class SplineXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = Quantity<XsGridData::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION SplineXsCalculator(XsGridData const& grid,
                                             Values const& values,
                                             size_type const order);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the cross section at the given index
    inline CELER_FUNCTION real_type operator[](size_type index) const;

    // Get the minimum energy
    CELER_FUNCTION Energy energy_min() const
    {
        return Energy(std::exp(loge_grid_.front()));
    }

    // Get the maximum energy
    CELER_FUNCTION Energy energy_max() const
    {
        return Energy(std::exp(loge_grid_.back()));
    }

  private:
    XsGridData const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;
    size_type order_;

    inline CELER_FUNCTION real_type get(size_type index) const;
    CELER_FORCEINLINE_FUNCTION real_type interpolate(real_type energy,
                                                     size_type low_idx,
                                                     size_type high_idx) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
SplineXsCalculator::SplineXsCalculator(XsGridData const& grid,
                                       Values const& values,
                                       size_type const order)
    : data_(grid), reals_(values), loge_grid_(data_.log_energy), order_(order)
{
    CELER_EXPECT(data_);
    CELER_ASSERT(grid.value.size() == data_.log_energy.size);

    // Order of interpolation must be smaller than the grid size for effective
    // spline interpolation. Max order is set to be clipped at minimum and
    // maximum energy indices so this may be unnecessary
    CELER_ASSERT(order < grid.value.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * If needed, we can add a "log(energy/MeV)" accessor if we constantly reuse
 * that value and don't want to repeat the `std::log` operation.
 */
CELER_FUNCTION real_type SplineXsCalculator::operator()(Energy energy) const
{
    real_type const loge = std::log(energy.value());

    auto calc_extrapolated = [this, &energy](size_type idx) {
        real_type result = this->get(idx);
        if (idx >= data_.prime_index)
        {
            result /= energy.value();
        }
        return result;
    };

    // Snap out-of-bounds values to closest grid points
    if (loge <= loge_grid_.front())
    {
        return calc_extrapolated(0);
    }
    if (loge >= loge_grid_.back())
    {
        return calc_extrapolated(loge_grid_.size() - 1);
    }

    // Locate the energy bin
    size_type lower_idx = loge_grid_.find(loge);
    CELER_ASSERT(lower_idx + 1 < loge_grid_.size());

    // Number of grid indexs away from the specified energy that need to be
    // checked in both directions
    size_type order_steps = order_ / 2 + 1;

    // True bounding indices of the grid that will be checked.
    // If the interpolation requests out-of-bounds indices, clip the
    // extents. This will reduce the order of the interpolation
    // TODO: instead of clipping the bounds, alter both the low and high
    // index to keep the range just shifted down

    size_type true_low_idx;
    if (lower_idx >= order_steps - 1)
    {
        true_low_idx = lower_idx - order_steps + 1;
    }
    else
    {
        true_low_idx = 0;
    }
    size_type true_high_idx
        = min(lower_idx + order_steps + 1, loge_grid_.size());

    if (order_ % 2 == 0)
    {
        // If the requested interpolation order is even, a direction must be
        // selected to interpolate to
        real_type low_dist = std::fabs(loge - loge_grid_[lower_idx]);
        real_type high_dist = std::fabs(loge_grid_[lower_idx + 1] - loge);

        if (true_high_idx - true_low_idx > order_ + 1)
        {
            // If we already clipped based on the bounding indices, don't clip
            // again
            if (low_dist > high_dist)
            {
                true_low_idx += 1;
            }
            else
            {
                true_high_idx -= 1;
            }
        }
    }

    return this->interpolate(energy.value(), true_low_idx, true_high_idx);
}

//---------------------------------------------------------------------------//
/*!
 * Get the cross section at the given index.
 */
CELER_FUNCTION real_type SplineXsCalculator::operator[](size_type index) const
{
    real_type energy = std::exp(loge_grid_[index]);
    real_type result = this->get(index);

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
CELER_FUNCTION real_type SplineXsCalculator::get(size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate the cross sections using spline.
 */
CELER_FUNCTION real_type SplineXsCalculator::interpolate(
    real_type energy, size_type low_idx, size_type high_idx) const
{
    CELER_EXPECT(high_idx <= loge_grid_.size());
    real_type result = 0;

    // Outer loop over indices for contributing to the result
    for (size_type outer_idx = low_idx; outer_idx < high_idx; ++outer_idx)
    {
        real_type outer_e = std::exp(loge_grid_[outer_idx]);
        real_type num = 1;
        real_type denom = 1;

        // Inner loop over indices for determining the weight
        for (size_type inner_idx = low_idx; inner_idx < high_idx; ++inner_idx)
        {
            // Don't contribute for inner and outer index the same
            if (inner_idx != outer_idx)
            {
                real_type inner_e = std::exp(loge_grid_[inner_idx]);
                num *= (energy - inner_e);
                denom *= (outer_e - inner_e);
            }
        }
        real_type weighted_value = (num / denom) * this->get(outer_idx);
        if (outer_idx >= data_.prime_index)
        {
            weighted_value /= outer_e;
        }

        result += weighted_value;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
