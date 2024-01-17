//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/TwodSubgridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"

#include "FindInterp.hh"
#include "NonuniformGrid.hh"
#include "TwodGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Do bilinear interpolation on a 2D grid with the x value preselected.
 *
 * This is usually not called directly but rather given as the return result of
 * the TwodGridCalculator.
 */
class TwodSubgridCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    using InterpT = FindInterp<real_type>;
    //!@}

  public:
    // Construct with grid data, backend values, and lower X data.
    inline CELER_FUNCTION TwodSubgridCalculator(TwodGridData const& grid,
                                                Values const& storage,
                                                InterpT x_loc);

    // Calculate the value at the given y coordinate
    inline CELER_FUNCTION real_type operator()(real_type y) const;

    //! Index of the preselected lower x value
    CELER_FORCEINLINE_FUNCTION size_type x_index() const
    {
        return x_loc_.index;
    }

    //! Fraction between the lower and upper x grid values
    CELER_FORCEINLINE_FUNCTION real_type x_fraction() const
    {
        return x_loc_.fraction;
    }

  private:
    TwodGridData const& grids_;
    Values const& storage_;
    InterpT const x_loc_;

    inline CELER_FUNCTION real_type at(size_type x_idx, size_type y_idx) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with grid data, backend values, and lower X data.
 *
 * This is typically constructed from a TwodGridCalculator. The interpolated x
 * location could be extended to allow a fractional value of 1 to support
 * interpolating on the highest value of the x grid.
 */
CELER_FUNCTION
TwodSubgridCalculator::TwodSubgridCalculator(TwodGridData const& grids,
                                             Values const& storage,
                                             InterpT x_loc)
    : grids_{grids}, storage_(storage), x_loc_(x_loc)
{
    CELER_EXPECT(grids);
    CELER_EXPECT(grids.values.back() < storage.size());
    CELER_EXPECT(x_loc.index + 1 < grids.x.size());
    CELER_EXPECT(x_loc.fraction >= 0 && x_loc_.fraction < 1);
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
    NonuniformGrid<real_type> const y_grid{grids_.y, storage_};
    CELER_EXPECT(y >= y_grid.front() && y < y_grid.back());

    InterpT const y_loc = find_interp(y_grid, y);
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
}  // namespace celeritas
