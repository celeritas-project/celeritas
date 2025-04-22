//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/data/Collection.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/SplineInterpolator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Quantity.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate range on a uniform log grid.
 *
 * \code
    RangeCalculator calc_range(xs_grid, xs_params.reals);
    real_type range = calc_range(particle);
   \endcode
 *
 * Below the minimum tabulated energy, the range is scaled:
 * \f[
    r = r_\mathrm{min} \sqrt{\frac{E}{E_\mathrm{min}}}
 * \f]
 *
 * \todo Construct with \c UniformGridRecord
 */
class RangeCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = RealQuantity<XsGridRecord::EnergyUnits>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from state-independent data
    inline CELER_FUNCTION
    RangeCalculator(XsGridRecord const& grid, Values const& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    UniformGridRecord const& data_;
    Values const& reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 *
 * Range tables should be uniform in energy, without extra scaling.
 */
CELER_FUNCTION
RangeCalculator::RangeCalculator(XsGridRecord const& grid, Values const& values)
    : data_(grid.lower), reals_(values)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(!grid.upper);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the range.
 */
CELER_FUNCTION real_type RangeCalculator::operator()(Energy energy) const
{
    CELER_ASSERT(energy > zero_quantity());
    UniformGrid loge_grid(data_.grid);
    real_type const loge = std::log(energy.value());

    if (loge <= loge_grid.front())
    {
        real_type result = this->get(0);
        // Scale by sqrt(E/Emin) = exp(.5 (log E - log Emin))
        result *= std::exp(real_type(.5) * (loge - loge_grid.front()));
        return result;
    }
    else if (loge >= loge_grid.back())
    {
        // Clip to highest range value
        return this->get(loge_grid.size() - 1);
    }

    // Locate the energy bin
    auto idx = loge_grid.find(loge);
    CELER_ASSERT(idx + 1 < loge_grid.size());

    real_type result;
    if (data_.derivative.empty())
    {
        // Interpolate *linearly* on energy
        result = LinearInterpolator<real_type>(
            {std::exp(loge_grid[idx]), this->get(idx)},
            {std::exp(loge_grid[idx + 1]), this->get(idx + 1)})(
            value_as<Energy>(energy));
    }
    else
    {
        // Use cubic spline interpolation
        real_type lower_deriv = reals_[data_.derivative[idx]];
        real_type upper_deriv = reals_[data_.derivative[idx + 1]];

        result = SplineInterpolator<real_type>(
            {std::exp(loge_grid[idx]), this->get(idx), lower_deriv},
            {std::exp(loge_grid[idx + 1]), this->get(idx + 1), upper_deriv})(
            value_as<Energy>(energy));
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw range data at a particular index.
 */
CELER_FUNCTION real_type RangeCalculator::get(size_type index) const
{
    CELER_EXPECT(index < reals_.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
