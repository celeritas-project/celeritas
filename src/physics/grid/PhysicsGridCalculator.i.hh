//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Interpolator.hh"
#include "physics/grid/UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from cross section data.
 */
CELER_FUNCTION
PhysicsGridCalculator::PhysicsGridCalculator(const XsGridPointers& data)
    : data_(data)
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * If needed, we can add a "log(energy/MeV)" accessor if we constantly reuse
 * that value and don't want to repeat the `std::log` operation.
 */
CELER_FUNCTION real_type PhysicsGridCalculator::operator()(Energy energy) const
{
    UniformGrid loge_grid(data_.log_energy);
    real_type   loge = std::log(energy.value());

    // Snap out-of-bounds values to closest grid points
    size_type lower_idx;
    real_type result;
    if (loge <= loge_grid.front())
    {
        lower_idx = 0;
        result    = data_.value.front();
    }
    else if (loge >= loge_grid.back())
    {
        lower_idx = data_.value.size() - 1;
        result    = data_.value.back();
    }
    else
    {
        // Locate the energy bin
        lower_idx = loge_grid.find(loge);
        CELER_ASSERT(lower_idx + 1 < data_.value.size());

        real_type upper_xs     = data_.value[lower_idx + 1];
        real_type upper_energy = std::exp(loge_grid[lower_idx + 1]);
        if (lower_idx + 1 == data_.prime_index)
        {
            // Cross section data for the upper point has *already* been scaled
            // by E -- undo the scaling.
            upper_xs /= upper_energy;
        }

        // Interpolate *linearly* on energy using the lower_idx data.
        LinearInterpolator<real_type> interpolate_xs(
            {std::exp(loge_grid[lower_idx]), data_.value[lower_idx]},
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
} // namespace celeritas
