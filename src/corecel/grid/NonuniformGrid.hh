//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/NonuniformGrid.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Algorithms.hh"

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
    //! \name Type aliases
    using size_type = ::celeritas::size_type;
    using value_type = T;
    using Values
        = Collection<value_type, Ownership::const_reference, MemSpace::native>;
    using SpanConstT = typename Values::SpanConstT;
    //!@}

  public:
    // Construct with data
    inline CELER_FUNCTION
    NonuniformGrid(ItemRange<value_type> const& values, Values const& data);

    // Construct with data (all values)
    explicit inline CELER_FUNCTION NonuniformGrid(Values const& data);

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
    SpanConstT data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data.
 */
template<class T>
CELER_FUNCTION
NonuniformGrid<T>::NonuniformGrid(ItemRange<value_type> const& values,
                                  Values const& data)
    : data_(data[values])
{
    CELER_EXPECT(data_.size() >= 2);
    CELER_EXPECT(data_.front() <= data_.back());  // Approximation for "sorted"
}

//---------------------------------------------------------------------------//
/*!
 * Construct with data (all values).
 */
template<class T>
CELER_FUNCTION NonuniformGrid<T>::NonuniformGrid(Values const& data)
    : data_(data[AllItems<value_type>{}])
{
    CELER_EXPECT(data_.size() >= 2);
    CELER_EXPECT(data_.front() <= data_.back());  // Approximation for "sorted"
}

//---------------------------------------------------------------------------//
/*!
 * Get the value at the given grid point.
 */
template<class T>
CELER_FUNCTION auto NonuniformGrid<T>::operator[](size_type i) const
    -> value_type
{
    CELER_EXPECT(i < data_.size());
    return data_[i];
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
template<class T>
CELER_FUNCTION size_type NonuniformGrid<T>::find(value_type value) const
{
    CELER_EXPECT(value >= this->front() && value < this->back());

    auto iter = celeritas::lower_bound(data_.begin(), data_.end(), value);
    CELER_ASSERT(iter != data_.end());

    if (value != *iter)
    {
        // Exactly on end grid point, or not on a grid point at all: move to
        // previous bin
        --iter;
    }

    return iter - data_.begin();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
