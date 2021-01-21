//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsGridCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Interpolator.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/grid/UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section from the particle state.
 *
 * This signature is here to allow for potential acceleration by precalculating
 * log(E)
 */
CELER_FUNCTION real_type
PhysicsGridCalculator::operator()(const ParticleTrackView& particle) const
{
    return (*this)(particle.energy());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 *
 * Assumes that the energy grid has the same units as particle.energy.
 *
 * XXX also this breaks if prime_energy != 1, since the
 * value of "xs" is actually xs*E above E' but the stored value at the lower
 * grid point is just xs.
 *
 * To fix that, we can change 'prime_energy' to 'prime_energy_index' since it
 * should always be on a grid point. If `bin == prime_energy_index`, then scale
 * the lower xs value by E. If bin >= prime_energy_index, scale the result by
 * 1/E.
 */
CELER_FUNCTION real_type PhysicsGridCalculator::operator()(Energy energy) const
{
    UniformGrid loge_grid(data_.log_energy);
    real_type   loge = std::log(energy.value());

    // Snap out-of-bounds values to closest grid points
    size_type bin;
    real_type result;
    if (loge <= loge_grid.front())
    {
        bin    = 0;
        result = data_.value.front();
    }
    else if (loge >= loge_grid.back())
    {
        bin    = data_.value.size();
        result = data_.value.back();
    }
    else
    {
        // Get the energy bin
        bin = loge_grid.find(loge);
        CELER_ASSERT(bin + 1 < data_.value.size());

        // Interpolate *linearly* on energy using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {std::exp(loge_grid[bin]), data_.value[bin]},
            {std::exp(loge_grid[bin + 1]), data_.value[bin + 1]});
        result = interpolate_xs(energy.value());
    }

    if (bin > data_.prime_index)
    {
        result /= energy.value();
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
