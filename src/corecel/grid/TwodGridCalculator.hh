//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/TwodGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"

#include "FindInterp.hh"
#include "TwodGridData.hh"
#include "TwodSubgridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and do bilinear interpolation on a nonuniform 2D grid of reals.
 *
 * Values should be node-centered, at the intersection of the two grids.
 *
 * \code
    TwodGridCalculator calc(grid, params.reals);
    real_type interpolated = calc({energy.value(), exit});
    // Or if the incident energy is reused...
    auto calc2 = calc(energy.value());
    interpolated = calc2(exit);
   \endcode
 */
class TwodGridCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Point = Array<real_type, 2>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with grid data and backend values
    inline CELER_FUNCTION
    TwodGridCalculator(TwodGridData const& grid, Values const& storage);

    // Calculate the value at the given x, y coordinates
    inline CELER_FUNCTION real_type operator()(Point const& xy) const;

    // Get an interpolator for calculating y values for a given x
    inline CELER_FUNCTION TwodSubgridCalculator operator()(real_type x) const;

  private:
    TwodGridData const& grids_;
    Values const& storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with grids and node-centered data.
 */
CELER_FUNCTION TwodGridCalculator::TwodGridCalculator(TwodGridData const& grids,
                                                      Values const& storage)
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
CELER_FUNCTION real_type TwodGridCalculator::operator()(Point const& inp) const
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
    NonuniformGrid<real_type> const x_grid{grids_.x, storage_};
    CELER_EXPECT(x >= x_grid.front() && x < x_grid.back());
    return {grids_, storage_, find_interp(x_grid, x)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
