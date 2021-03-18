//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodGridCalculator.i.hh
//---------------------------------------------------------------------------//
#include "NonuniformGrid.hh"
#include "detail/FindInterp.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with grids and node-centered data.
 */
CELER_FUNCTION TwodGridCalculator::TwodGridCalculator(const TwodGridData& grids,
                                                      const Values& storage)
    : grids_{grids}, storage_(storage)
{
    CELER_EXPECT(grids);
    CELER_EXPECT(grids.values.back() < storage.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the value at the given (x, y) coordinates.
 *
 * The coordinates must be inside \f$0 <= x < x_\mathrm{max}\f$ and
 * \f$0 <= y < y_\mathrm{max}\f$.
 *
 * \todo We may need to add logic inside the axis loop to account for points
 * outside the grid.
 */
CELER_FUNCTION real_type TwodGridCalculator::operator()(const Point& inp) const
{
    return (*this)(inp[0])(inp[1]);
}

//---------------------------------------------------------------------------//
/*!
 * Get an interpolator for a preselected x value.
 */
CELER_FUNCTION TwodSubgridCalculator
TwodGridCalculator::operator()(real_type x) const
{
    const NonuniformGrid<real_type> x_grid{grids_.x, storage_};
    CELER_EXPECT(x >= x_grid.front() && x < x_grid.back());
    return {grids_, storage_, detail::find_interp(x_grid, x)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
