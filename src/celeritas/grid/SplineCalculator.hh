//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"

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
    SplineCalculator calc_xs(xs_grid, xs_params.reals);
    real_type xs = calc_xs(particle);
   \endcode
 */
class SplineCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    SplineCalculator(UniformGridRecord const& grid, Values const& reals);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    // Get the value at the given index
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
    UniformGridRecord const& data_;
    Values const& reals_;
    UniformGrid loge_grid_;

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
SplineCalculator::SplineCalculator(UniformGridRecord const& grid,
                                   Values const& reals)
    : data_(grid), reals_(reals), loge_grid_(data_.grid)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * If needed, we can add a "log(energy/MeV)" accessor if we constantly reuse
 * that value and don't want to repeat the `std::log` operation.
 */
CELER_FUNCTION real_type SplineCalculator::operator()(Energy energy) const
{
    real_type const loge = std::log(energy.value());

    // Snap out-of-bounds values to closest grid points
    if (loge <= loge_grid_.front())
    {
        return (*this)[0];
    }
    if (loge >= loge_grid_.back())
    {
        return (*this)[loge_grid_.size() - 1];
    }

    // Locate the energy bin
    size_type lower_idx = loge_grid_.find(loge);
    CELER_ASSERT(lower_idx + 1 < loge_grid_.size());

    // Number of grid indices away from the specified energy that need to be
    // checked in both directions
    size_type order_steps = data_.spline_order / 2 + 1;

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

    if (data_.spline_order % 2 == 0)
    {
        // If the requested interpolation order is even, a direction must be
        // selected to interpolate to
        real_type low_dist = std::fabs(loge - loge_grid_[lower_idx]);
        real_type high_dist = std::fabs(loge_grid_[lower_idx + 1] - loge);

        if (true_high_idx - true_low_idx > data_.spline_order + 1)
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
 * Get the tabulated value at the given index.
 */
CELER_FUNCTION real_type SplineCalculator::operator[](size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Interpolate the value using spline.
 */
CELER_FUNCTION real_type SplineCalculator::interpolate(real_type energy,
                                                       size_type low_idx,
                                                       size_type high_idx) const
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
        result += (num / denom) * (*this)[outer_idx];
    }
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
