//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/data/Collection.hh"
#include "corecel/grid/Interpolator.hh"
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
 */
class RangeCalculator
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
    inline CELER_FUNCTION
    RangeCalculator(XsGridData const& grid, Values const& values);

    // Find and interpolate from the energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    XsGridData const& data_;
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
RangeCalculator::RangeCalculator(XsGridData const& grid, Values const& values)
    : data_(grid), reals_(values)
{
    CELER_EXPECT(data_);
    CELER_EXPECT(data_.prime_index == XsGridData::no_scaling());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the range.
 */
CELER_FUNCTION real_type RangeCalculator::operator()(Energy energy) const
{
    CELER_ASSERT(energy > zero_quantity());
    UniformGrid loge_grid(data_.log_energy);
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

    // Interpolate *linearly* on energy
    LinearInterpolator<real_type> interpolate_xs(
        {std::exp(loge_grid[idx]), this->get(idx)},
        {std::exp(loge_grid[idx + 1]), this->get(idx + 1)});
    return interpolate_xs(energy.value());
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
