//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreXsCalculator.i.hh
//---------------------------------------------------------------------------//
#include <cmath>
#include "base/Macros.hh"
#include "physics/grid/Interpolator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from state-independent data.
 */
CELER_FUNCTION
LivermoreXsCalculator::LivermoreXsCalculator(const LivermoreValueGrid& data)
    : data_(data)
{
    CELER_EXPECT(data_.energy.size() > 0);
    CELER_EXPECT(data_.xs.size() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section.
 */
CELER_FUNCTION real_type
LivermoreXsCalculator::operator()(const real_type energy) const
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
        // Get the energy bin.
        // TODO: Should do a binary search, but this class is just a
        // placeholder anyway.
        auto bin = data_.energy.size();
        while (data_.energy[--bin] >= energy) {}
        CELER_ASSERT(bin + 1 < data_.xs.size());

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
