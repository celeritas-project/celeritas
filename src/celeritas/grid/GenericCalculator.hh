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
 * Find and interpolate values on a nonuniform grid.
 *
 * Out-of-bounds values are snapped to the closest grid points.
 */
class GenericCalculator
{
  public:
    //@{
    //! Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //@}

  public:
    // Construct from grid data and backend values
    inline CELER_FUNCTION
    GenericCalculator(GenericGridRecord const& grid, Values const& values);

    // Find and interpolate the y value from the given x value
    inline CELER_FUNCTION real_type operator()(real_type x) const;

    // Get the tabulated y value at a particular index.
    inline CELER_FUNCTION real_type operator[](size_type index) const;

    // Get the tabulated x values
    inline CELER_FUNCTION NonuniformGrid<real_type> const& grid() const;

  private:
    Values const& reals_;
    NonuniformGrid<real_type> x_grid_;
    ItemRange<real_type> value_index_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend values.
 */
CELER_FUNCTION
GenericCalculator::GenericCalculator(GenericGridRecord const& grid,
                                     Values const& values)
    : reals_{values}, x_grid_{grid.grid, reals_}, value_index_{grid.value}
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the y value at the given x value.
 */
CELER_FUNCTION real_type GenericCalculator::operator()(real_type x) const
{
    // Snap out-of-bounds values to closest grid points
    size_type lower_idx;
    real_type result;
    if (x <= x_grid_.front())
    {
        lower_idx = 0;
        result = (*this)[lower_idx];
    }
    else if (x >= x_grid_.back())
    {
        lower_idx = x_grid_.size() - 1;
        result = (*this)[lower_idx];
    }
    else
    {
        // Locate the x bin
        lower_idx = x_grid_.find(x);
        CELER_ASSERT(lower_idx + 1 < x_grid_.size());

        // Interpolate *linearly* on x using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {x_grid_[lower_idx], (*this)[lower_idx]},
            {x_grid_[lower_idx + 1], (*this)[lower_idx + 1]});
        result = interpolate_xs(x);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the tabulated y value at a particular index.
 */
CELER_FUNCTION real_type GenericCalculator::operator[](size_type index) const
{
    CELER_EXPECT(index < value_index_.size());
    return reals_[value_index_[index]];
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
}  // namespace celeritas
