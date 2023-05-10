//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/NonuniformGrid.hh"

#include "XsGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross sections on a nonuniform grid.
 */
class GenericXsCalculator
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
    GenericXsCalculator(GenericGridData const& grid, Values const& values);

    // Find and interpolate the cross section from the given energy
    inline CELER_FUNCTION real_type operator()(const real_type energy) const;

  private:
    GenericGridData const& data_;
    Values const& reals_;

    CELER_FORCEINLINE_FUNCTION real_type get(size_type index) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from grid data and backend values.
 */
CELER_FUNCTION
GenericXsCalculator::GenericXsCalculator(GenericGridData const& grid,
                                         Values const& values)
    : data_(grid), reals_(values)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cross section at the given energy.
 */
CELER_FUNCTION real_type
GenericXsCalculator::operator()(const real_type energy) const
{
    NonuniformGrid<real_type> const energy_grid(data_.grid, reals_);

    // Snap out-of-bounds values to closest grid points
    size_type lower_idx;
    real_type result;
    if (energy <= energy_grid.front())
    {
        lower_idx = 0;
        result = this->get(lower_idx);
    }
    else if (energy >= energy_grid.back())
    {
        lower_idx = energy_grid.size() - 1;
        result = this->get(lower_idx);
    }
    else
    {
        // Locate the energy bin
        lower_idx = energy_grid.find(energy);
        CELER_ASSERT(lower_idx + 1 < energy_grid.size());

        // Interpolate *linearly* on energy using the bin data.
        LinearInterpolator<real_type> interpolate_xs(
            {energy_grid[lower_idx], this->get(lower_idx)},
            {energy_grid[lower_idx + 1], this->get(lower_idx + 1)});
        result = interpolate_xs(energy);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw cross section data at a particular index.
 */
CELER_FUNCTION real_type GenericXsCalculator::get(size_type index) const
{
    CELER_EXPECT(index < data_.value.size());
    return reals_[data_.value[index]];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
