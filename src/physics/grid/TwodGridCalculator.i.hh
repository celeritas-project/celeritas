//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodGridCalculator.i.hh
//---------------------------------------------------------------------------//
#include "base/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with grids and node-centered data.
 */
CELER_FUNCTION TwodGridCalculator::TwodGridCalculator(const TwodGridData& grids,
                                                      const Values& storage)
    : grids_{GridT{grids.x, storage}, GridT{grids.y, storage}}
    , value_offset_{grids.values.front().get()}
    , storage_(storage)
{
    CELER_EXPECT(grids);
    CELER_EXPECT(grids.values.back() < storage.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the value at the given (x, y) coordinates.
 *
 * \todo We may need to add logic inside the axis loop to account for points
 * outside the grid.
 */
CELER_FUNCTION real_type TwodGridCalculator::operator()(const Point& inp) const
{
    // Find X/Y lower grid points and fraction between those points and upper
    // neighbor along each axis.
    size_type idx[2];
    real_type frac[2];
    for (int ax : {X, Y})
    {
        CELER_EXPECT(inp[ax] >= grids_[ax].front()
                     && inp[ax] < grids_[ax].back());
        idx[ax]                   = grids_[ax].find(inp[ax]);
        const real_type lower_val = grids_[ax][idx[ax]];
        const real_type upper_val = grids_[ax][idx[ax] + 1];
        frac[ax] = (inp[ax] - lower_val) / (upper_val - lower_val);
    }

    // clang-format off
    return   (1 - frac[X]) * (1 - frac[Y]) * this->at(idx[X]    , idx[Y]    )
           + (1 - frac[X]) * (    frac[Y]) * this->at(idx[X]    , idx[Y] + 1)
           + (    frac[X]) * (1 - frac[Y]) * this->at(idx[X] + 1, idx[Y]    )
           + (    frac[X]) * (    frac[Y]) * this->at(idx[X] + 1, idx[Y] + 1);
    // clang-format on
}

//---------------------------------------------------------------------------//
/*!
 * Get the value at the specified x/y coordinate.
 *
 * NOTE: this must match TwodGridData::index.
 */
CELER_FUNCTION real_type TwodGridCalculator::at(size_type x_idx,
                                                size_type y_idx) const
{
    using IdReal = ItemId<real_type>;
    return storage_[IdReal{x_idx * grids_[Y].size() + y_idx + value_offset_}];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
