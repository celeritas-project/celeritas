//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenericXsCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Macros.hh"
#include "Interpolator.hh"
#include "NonuniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend values.
 */
CELER_FUNCTION
GenericXsCalculator::GenericXsCalculator(const GenericGridData& grid,
                                         const Values&          values)
    : data_(grid), reals_(values)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section at the given energy.
 */
CELER_FUNCTION real_type
GenericXsCalculator::operator()(const real_type energy) const
{
    const NonuniformGrid<real_type> energy_grid(data_.grid, reals_);

    // Snap out-of-bounds values to closest grid points
    size_type lower_idx;
    real_type result;
    if (energy <= energy_grid.front())
    {
        lower_idx = 0;
        result    = this->get(lower_idx);
    }
    else if (energy >= energy_grid.back())
    {
        lower_idx = energy_grid.size() - 1;
        result    = this->get(lower_idx);
    }
    else
    {
        // Locate the energy bin
        lower_idx = energy_grid.find(energy);
        CELER_ASSERT(lower_idx + 1 < energy_grid.size());

        // Interpolate *linearly* on energy using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {energy_grid[lower_idx], this->get(lower_idx)},
            {energy_grid[lower_idx + 1], this->get(lower_idx + 1)});
        result = interpolate_xs(energy);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw cross section data at a particular index.
 */
CELER_FUNCTION real_type GenericXsCalculator::get(size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
