//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockXsCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Interpolator.hh"
#include "base/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from state-independent data.
 */
CELER_FUNCTION XsCalculator::XsCalculator(const ValueGrid& data) : data_(data)
{
    REQUIRE(data_.energy.size() > 0);
    REQUIRE(data_.xs.size() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 */
CELER_FUNCTION real_type XsCalculator::operator()(const real_type energy) const
{
    // Snap out-of-bounds values to closest grid points
    real_type result;
    if (energy <= data_.energy.front())
    {
        result = data_.xs.front();
    }
    else if (energy >= data_.energy.back())
    {
        result = data_.xs.back();
    }
    else
    {
        // Get the energy bin
        // TODO: binary search
        auto bin = data_.energy.size();
        while (data_.energy[--bin] > energy) {}
        CHECK(bin + 1 < data_.xs.size());

        // Interpolate *linearly* on energy using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {data_.energy[bin], data_.xs[bin]},
            {data_.energy[bin + 1], data_.xs[bin + 1]});
        result = interpolate_xs(energy);
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
