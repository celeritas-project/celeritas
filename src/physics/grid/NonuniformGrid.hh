//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NonuniformGrid.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with a nonuniform grid of increasing values.
 *
 * This should have the same interface (aside from constructor) as
 * UniformGrid.
 */
template<class T>
class NonuniformGrid
{
  public:
    //!@{
    //! Type aliases
    using size_type  = ::celeritas::size_type;
    using value_type = T;
    using Values
        = Collection<value_type, Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with data
    explicit inline CELER_FUNCTION
    NonuniformGrid(const ItemRange<value_type>& values, const Values& data);

    // Construct with data (all values)
    explicit inline CELER_FUNCTION NonuniformGrid(const Values& data);

    //! Number of grid points
    CELER_FORCEINLINE_FUNCTION size_type size() const { return data_.size(); }

    //! Minimum/first value
    CELER_FORCEINLINE_FUNCTION value_type front() const
    {
        return data_.front();
    }

    //! Maximum/last value
    CELER_FORCEINLINE_FUNCTION value_type back() const { return data_.back(); }

    // Calculate the value at the given grid point
    inline CELER_FUNCTION value_type operator[](size_type i) const;

    // Find the index of the given value (*must* be in bounds)
    inline CELER_FUNCTION size_type find(value_type value) const;

  private:
    // TODO: change backend for effiency if needeed
    Span<const value_type> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "NonuniformGrid.i.hh"
