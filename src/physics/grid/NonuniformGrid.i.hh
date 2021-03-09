//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NonuniformGrid.i.hh
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with data.
 */
template<class T>
CELER_FUNCTION
NonuniformGrid<T>::NonuniformGrid(const ItemRange<value_type>& values,
                                  const Values&                data)
    : data_(data[values])
{
    CELER_EXPECT(data_.size() >= 2);
    CELER_EXPECT(data_.front() <= data_.back()); // Approximation for "sorted"
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
} // namespace celeritas
