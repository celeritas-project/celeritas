//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/NonuniformGrid.hh"

#include "GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate real numbers on a nonuniform grid.
 *
 * The end points of the grid are extrapolated outward as constant values.
 */
class GenericCalculator
{
  public:
    //@{
    //! Type aliases
    using Reals
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    using Grid = NonuniformGrid<real_type>;
    //@}

  public:
    // Construct by *inverting* a monotonicially increasing generic grid
    static inline CELER_FUNCTION GenericCalculator
    from_inverse(GenericGridRecord const& grid, Reals const& storage);

    // Construct from grid data and backend storage
    inline CELER_FUNCTION
    GenericCalculator(GenericGridRecord const& grid, Reals const& storage);

    // Find and interpolate the y value from the given x value
    inline CELER_FUNCTION real_type operator()(real_type x) const;

    // Get the tabulated y value at a particular index.
    inline CELER_FUNCTION real_type operator[](size_type index) const;

    // Get the tabulated x grid
    inline CELER_FUNCTION Grid const& grid() const;

    // Make a calculator with x and y flipped
    inline CELER_FUNCTION GenericCalculator make_inverse() const;

  private:
    //// TYPES ////

    using RealIds = ItemRange<real_type>;

    //// DATA ////

    Reals const& reals_;
    Grid x_grid_;
    RealIds y_offset_;

    //// HELPERS ////

    // Private constructor implementation
    inline CELER_FUNCTION
    GenericCalculator(Reals const& storage, RealIds x_grid, RealIds y_grid);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct by \em inverting a monotonicially increasing generic grid.
 */
CELER_FUNCTION GenericCalculator GenericCalculator::from_inverse(
    GenericGridRecord const& grid, Reals const& storage)
{
    return GenericCalculator{storage, grid.value, grid.grid};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend storage.
 */
CELER_FUNCTION
GenericCalculator::GenericCalculator(GenericGridRecord const& grid,
                                     Reals const& storage)
    : GenericCalculator{storage, grid.grid, grid.value}
{
    CELER_EXPECT(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the y value at the given x value.
 */
CELER_FUNCTION real_type GenericCalculator::operator()(real_type x) const
{
    // Snap out-of-bounds values to closest grid points
    if (x <= x_grid_.front())
    {
        return (*this)[0];
    }
    if (x >= x_grid_.back())
    {
        return (*this)[x_grid_.size() - 1];
    }

    // Locate the x bin
    size_type lower_idx = x_grid_.find(x);
    CELER_ASSERT(lower_idx + 1 < x_grid_.size());

    // Interpolate *linearly* on x using the bin data.
    LinearInterpolator<real_type> interpolate_xs(
        {x_grid_[lower_idx], (*this)[lower_idx]},
        {x_grid_[lower_idx + 1], (*this)[lower_idx + 1]});
    return interpolate_xs(x);
}

//---------------------------------------------------------------------------//
/*!
 * Get the tabulated y value at a particular index.
 */
CELER_FUNCTION real_type GenericCalculator::operator[](size_type index) const
{
    CELER_EXPECT(index < y_offset_.size());
    return reals_[y_offset_[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Get the tabulated x values.
 */
CELER_FORCEINLINE_FUNCTION NonuniformGrid<real_type> const&
GenericCalculator::grid() const
{
    return x_grid_;
}

//---------------------------------------------------------------------------//
/*!
 * Make a calculator with x and y flipped.
 *
 * \pre The y values must be monotonic increasing.
 */
CELER_FUNCTION GenericCalculator GenericCalculator::make_inverse() const
{
    return GenericCalculator{reals_, y_offset_, x_grid_.offset()};
}

//---------------------------------------------------------------------------//
// PRIVATE HELPERS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend storage.
 */
CELER_FUNCTION
GenericCalculator::GenericCalculator(Reals const& storage,
                                     RealIds x_grid,
                                     RealIds y_grid)
    : reals_{storage}, x_grid_{x_grid, reals_}, y_offset_{y_grid}
{
    CELER_EXPECT(!x_grid.empty() && x_grid.size() == y_grid.size());
    CELER_EXPECT(*x_grid.end() <= reals_.size()
                 && *y_grid.end() <= reals_.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
