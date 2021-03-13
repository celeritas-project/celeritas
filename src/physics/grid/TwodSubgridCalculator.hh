//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TwodSubgridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "TwodGridInterface.hh"
#include "detail/FindInterp.hh"

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
    //! Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::native>;
    using InterpT = detail::FindInterp<real_type>;
    //!@}

  public:
    // Construct with grid data, backend values, and lower X data.
    inline TwodSubgridCalculator(const TwodGridData& grid,
                                 const Values&       storage,
                                 InterpT             x_loc);

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
    const TwodGridData& grids_;
    const Values&       storage_;
    InterpT             x_loc_;

    inline CELER_FUNCTION real_type at(size_type x_idx, size_type y_idx) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "TwodSubgridCalculator.i.hh"
