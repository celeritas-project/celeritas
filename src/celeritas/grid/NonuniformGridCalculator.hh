//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/grid/SplineInterpolator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate real numbers on a nonuniform grid.
 *
 * The end points of the grid are extrapolated outward as constant values.
 *
 * \todo Template on value type and/or units?
 */
class NonuniformGridCalculator
{
  public:
    //@{
    //! Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    using Grid = NonuniformGrid<real_type>;
    //@}

  public:
    // Construct by *inverting* a monotonicially increasing generic grid
    static inline CELER_FUNCTION NonuniformGridCalculator
    from_inverse(NonuniformGridRecord const& grid, Values const& reals);

    // Construct from grid data and backend storage
    inline CELER_FUNCTION
    NonuniformGridCalculator(NonuniformGridRecord const& grid,
                             Values const& reals);

    // Find and interpolate the y value from the given x value
    inline CELER_FUNCTION real_type operator()(real_type x) const;

    // Get the tabulated y value at a particular index.
    inline CELER_FUNCTION real_type operator[](size_type index) const;

    // Get the tabulated x grid
    inline CELER_FUNCTION Grid const& grid() const;

    // Make a calculator with x and y flipped
    inline CELER_FUNCTION NonuniformGridCalculator make_inverse() const;

  private:
    //// TYPES ////

    using RealIds = ItemRange<real_type>;

    //// DATA ////

    Values const& reals_;
    Grid x_grid_;
    RealIds y_offset_;
    RealIds deriv_offset_;

    //// HELPER FUNCTIONS ////

    // Private constructor implementation
    inline CELER_FUNCTION NonuniformGridCalculator(Values const& reals,
                                                   RealIds x_grid,
                                                   RealIds y_grid,
                                                   RealIds deriv);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct by \em inverting a monotonicially increasing generic grid.
 */
CELER_FUNCTION NonuniformGridCalculator NonuniformGridCalculator::from_inverse(
    NonuniformGridRecord const& grid, Values const& reals)
{
    CELER_EXPECT(grid.derivative.empty());
    return NonuniformGridCalculator{reals, grid.value, grid.grid, {}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend storage.
 */
CELER_FUNCTION
NonuniformGridCalculator::NonuniformGridCalculator(
    NonuniformGridRecord const& grid, Values const& reals)
    : NonuniformGridCalculator{reals, grid.grid, grid.value, grid.derivative}
{
    CELER_EXPECT(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the y value at the given x value.
 */
CELER_FUNCTION real_type NonuniformGridCalculator::operator()(real_type x) const
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

    real_type result;
    if (deriv_offset_.empty())
    {
        // Interpolate *linearly* on x using the bin data.
        result = LinearInterpolator<real_type>(
            {x_grid_[lower_idx], (*this)[lower_idx]},
            {x_grid_[lower_idx + 1], (*this)[lower_idx + 1]})(x);
    }
    else
    {
        // Use cubic spline interpolation
        real_type lower_deriv = reals_[deriv_offset_[lower_idx]];
        real_type upper_deriv = reals_[deriv_offset_[lower_idx + 1]];

        result = SplineInterpolator<real_type>(
            {x_grid_[lower_idx], (*this)[lower_idx], lower_deriv},
            {x_grid_[lower_idx + 1], (*this)[lower_idx + 1], upper_deriv})(x);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the tabulated y value at a particular index.
 */
CELER_FUNCTION real_type NonuniformGridCalculator::operator[](size_type index) const
{
    CELER_EXPECT(index < y_offset_.size());
    return reals_[y_offset_[index]];
}

//---------------------------------------------------------------------------//
/*!
 * Get the tabulated x values.
 */
CELER_FORCEINLINE_FUNCTION NonuniformGrid<real_type> const&
NonuniformGridCalculator::grid() const
{
    return x_grid_;
}

//---------------------------------------------------------------------------//
/*!
 * Make a calculator with x and y flipped.
 *
 * \pre The y values must be monotonic increasing.
 *
 * \note Spline interpolation will not be used on an inverse grid.
 */
CELER_FUNCTION NonuniformGridCalculator
NonuniformGridCalculator::make_inverse() const
{
    CELER_EXPECT(deriv_offset_.empty());
    return NonuniformGridCalculator{reals_, y_offset_, x_grid_.offset(), {}};
}

//---------------------------------------------------------------------------//
// PRIVATE HELPERS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend storage.
 */
CELER_FUNCTION
NonuniformGridCalculator::NonuniformGridCalculator(Values const& reals,
                                                   RealIds x_grid,
                                                   RealIds y_grid,
                                                   RealIds deriv)
    : reals_{reals}
    , x_grid_{x_grid, reals_}
    , y_offset_{y_grid}
    , deriv_offset_(deriv)
{
    CELER_EXPECT(!x_grid.empty() && x_grid.size() == y_grid.size());
    CELER_EXPECT(*x_grid.end() <= reals_.size()
                 && *y_grid.end() <= reals_.size());
    CELER_EXPECT(deriv.empty() || deriv.size() == x_grid.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
