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
        CHECK(bin + 1 < data_.xs.size());

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
