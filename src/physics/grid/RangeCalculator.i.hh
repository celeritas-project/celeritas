//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RangeCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "Interpolator.hh"
#include "physics/grid/UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 *
 * Range tables should be uniform in energy, without extra scaling.
 */
CELER_FUNCTION
RangeCalculator::RangeCalculator(const XsGridData& grid, const Values& values)
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
    UniformGrid     loge_grid(data_.log_energy);
    const real_type loge = std::log(energy.value());

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
} // namespace celeritas
