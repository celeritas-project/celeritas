//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsArrayCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Interpolator.hh"
#include "base/UniformGrid.hh"

namespace celeritas
{
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
CELER_FUNCTION real_type
PhysicsArrayCalculator::operator()(const ParticleTrackView& particle) const
{
    UniformGrid     loge_grid(data_.log_energy);
    const real_type energy = particle.energy().value();
    real_type       loge   = std::log(energy);

    // Snap out-of-bounds values to closest grid points
    real_type result;
    if (loge <= loge_grid.front())
    {
        result = data_.xs.front();
    }
    else if (loge >= loge_grid.back())
    {
        result = data_.xs.back();
    }
    else
    {
        // Get the energy bin
        auto bin = loge_grid.find(loge);
        CELER_ASSERT(bin + 1 < data_.xs.size());

        // Interpolate *linearly* on energy using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {std::exp(loge_grid[bin]), data_.xs[bin]},
            {std::exp(loge_grid[bin + 1]), data_.xs[bin + 1]});
        result = interpolate_xs(energy);
    }

    if (energy > data_.prime_energy)
    {
        result /= energy;
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
