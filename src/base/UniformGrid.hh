//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformGrid.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with a uniform grid of values.
 *
 * This simple class is used by physics vectors and classes that need to do
 * lookups on a uniform grid.
 */
class UniformGrid
{
  public:
    //@{
    //! Type aliases
    using size_type  = ::celeritas::size_type;
    using value_type = ::celeritas::real_type;
    //@}

    struct Params
    {
        size_type  size;  //!< Number of grid edges/points
        value_type front; //!< Value of first grid point
        value_type delta; //!< Grid cell width

        //! Whether the struct is in an assigned state
        CELER_FUNCTION operator bool() const { return size > 1; }
    };

  public:
    // Construct with data
    explicit inline CELER_FUNCTION UniformGrid(const Params& data);

    // Number of grid points
    inline CELER_FUNCTION size_type size() const;

    //! Minimum/first value
    CELER_FUNCTION value_type front() const { return data_.front; }

    // Maximum/last value
    inline CELER_FUNCTION value_type back() const;

    // Calculate the value at the given grid point
    inline CELER_FUNCTION value_type operator[](size_type i) const;

    // Find the index of the given value (*must* be in bounds)
    inline CELER_FUNCTION size_type find(value_type value) const;

    //! Get the params used to construct this class
    CELER_FUNCTION const Params& params() const { return data_; }

  private:
    Params data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "UniformGrid.i.hh"
