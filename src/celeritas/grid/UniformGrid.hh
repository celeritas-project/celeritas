//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGrid.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "UniformGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with a uniform grid of increasing values.
 *
 * This simple class is used by physics vectors and classes that need to do
 * lookups on a uniform grid.
 */
class UniformGrid
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = ::celeritas::size_type;
    using value_type = ::celeritas::real_type;
    //!@}

  public:
    // Construct with data
    explicit inline CELER_FUNCTION UniformGrid(UniformGridData const& data);

    //! Number of grid points
    CELER_FORCEINLINE_FUNCTION size_type size() const { return data_.size; }

    //! Minimum/first value
    CELER_FORCEINLINE_FUNCTION value_type front() const { return data_.front; }

    //! Maximum/last value
    CELER_FORCEINLINE_FUNCTION value_type back() const { return data_.back; }

    // Calculate the value at the given grid point
    inline CELER_FUNCTION value_type operator[](size_type i) const;

    // Find the index of the given value (*must* be in bounds)
    inline CELER_FUNCTION size_type find(value_type value) const;

    //! Get the data used to construct this class
    CELER_FUNCTION UniformGridData const& data() const { return data_; }

  private:
    UniformGridData const& data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data.
 */
CELER_FUNCTION
UniformGrid::UniformGrid(UniformGridData const& data) : data_(data)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the value at the given grid point.
 */
CELER_FUNCTION auto UniformGrid::operator[](size_type i) const -> value_type
{
    CELER_EXPECT(i < data_.size);
    return data_.front + data_.delta * i;
}

//---------------------------------------------------------------------------//
/*!
 * Find the value bin such that data[result] <= value < data[result + 1].
 *
 * The given value *must* be in range, because out-of-bounds values usually
 * require different treatment (e.g. clipping to the boundary values rather
 * than interpolating). It's easier to test the exceptional cases (final grid
 * point) outside of the grid view.
 */
CELER_FUNCTION size_type UniformGrid::find(value_type value) const
{
    CELER_EXPECT(value >= this->front() && value < this->back());
    auto bin = static_cast<size_type>((value - data_.front) / data_.delta);
    CELER_ENSURE(bin + 1 < this->size());
    return bin;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
