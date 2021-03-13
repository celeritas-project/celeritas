//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodSubgridCalculator.i.hh
//---------------------------------------------------------------------------//
#include "NonuniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION
TwodSubgridCalculator::TwodSubgridCalculator(const TwodGridData& grids,
                                             const Values&       storage,
                                             InterpT             x_loc)
    : grids_{grids}, storage_(storage), x_loc_(x_loc)
{
    CELER_EXPECT(grids);
    CELER_EXPECT(grids.values.back() < storage.size());
    CELER_EXPECT(x_loc.index + 1 < grids.x.size());
    CELER_EXPECT(x_loc.fraction >= 0 && x_loc_.fraction <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the value at the given y coordinate for preselected x.
 *
 * This uses *bilinear* interpolation and and therefore exactly represents
 * functions that are a linear combination of 1, x, y, and xy.
 */
CELER_FUNCTION real_type TwodSubgridCalculator::operator()(real_type y) const
{
    const NonuniformGrid<real_type> y_grid{grids_.y, storage_};
    CELER_EXPECT(y >= y_grid.front() && y < y_grid.back());

    const InterpT y_loc = detail::find_interp(y_grid, y);
    auto at_corner = [this, y_loc](size_type xo, size_type yo) -> real_type {
        return this->at(x_loc_.index + xo, y_loc.index + yo);
    };

    return (1 - x_loc_.fraction)
               * ((1 - y_loc.fraction) * at_corner(0, 0)
                  + (y_loc.fraction) * at_corner(0, 1))
           + (x_loc_.fraction)
                 * ((1 - y_loc.fraction) * at_corner(1, 0)
                    + (y_loc.fraction) * at_corner(1, 1));
}

//---------------------------------------------------------------------------//
/*!
 * Get the value at the specified x/y coordinate.
 *
 * NOTE: this must match TwodGridData::index.
 */
CELER_FUNCTION real_type TwodSubgridCalculator::at(size_type x_idx,
                                                   size_type y_idx) const
{
    return storage_[grids_.at(x_idx, y_idx)];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
