//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "TwodGridInterface.hh"
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
    //! Type aliases
    using Point = Array<real_type, 2>;
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with grid data and backend values
    inline CELER_FUNCTION
    TwodGridCalculator(const TwodGridData& grid, const Values& storage);

    // Calculate the value at the given x, y coordinates
    inline CELER_FUNCTION real_type operator()(const Point& xy) const;

    // Get an interpolator for calculating y values for a given x
    inline CELER_FUNCTION TwodSubgridCalculator operator()(real_type x) const;

  private:
    const TwodGridData& grids_;
    const Values&   storage_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "TwodGridCalculator.i.hh"
