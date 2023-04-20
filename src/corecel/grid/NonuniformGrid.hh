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
    using Data = Span<value_type const>;

    struct result_type
    {
        size_type cell;  //!< [0, span.size())
        bool on_edge;

        bool operator==(result_type const& other) const
        {
            return cell == other.cell && on_edge == other.on_edge;
        }
    };

    //!@}

  public:
    // Construct from a range into supplied values
    inline CELER_FUNCTION
    NonuniformGrid(ItemRange<value_type> const& range, Values const& values);

    // Construct with all values
    explicit inline CELER_FUNCTION NonuniformGrid(Values const& values);

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

    // Find the index of the given value (must be in bounds)
    inline CELER_FUNCTION size_type find(value_type value) const;

    // Find the index (must be in bounds), and whether the value is on an edge
    inline CELER_FUNCTION result_type find_edge(value_type value) const;

  private:
    // TODO: change backend for effiency if needeed
    Data data_;

  private:
    //// HELPER METHODS ////

    // Return iterator for bin corresponding to value (must be in bounds)
    inline CELER_FUNCTION typename Data::iterator
    find_impl(value_type value) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a range into supplied values.
 */
template<class T>
CELER_FUNCTION
NonuniformGrid<T>::NonuniformGrid(ItemRange<value_type> const& range,
                                  Values const& values)
    : data_(values[range])
{
    CELER_EXPECT(data_.size() >= 2);
    CELER_EXPECT(data_.front() <= data_.back());  // Approximation for
                                                  // "sorted"
}

//---------------------------------------------------------------------------//
/*!
 * Construct with all values.
 */
template<class T>
CELER_FUNCTION NonuniformGrid<T>::NonuniformGrid(Values const& values)
    : data_(values[AllItems<value_type>{}])
{
    CELER_EXPECT(data_.size() >= 2);
    CELER_EXPECT(data_.front() <= data_.back());  // Approximation for
                                                  // "sorted"
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
 * Find the index of the given value (must be in bounds)
 *
 * Find the value bin such that values[result] <= value < values[result +
 * 1]. The given value *must* be in range, because out-of-bounds values usually
 * require different treatment (e.g. clipping to the boundary values rather
 * than interpolating). It's easier to test the exceptional cases (final
 * grid point) outside of the grid view.
 */
template<class T>
CELER_FUNCTION size_type NonuniformGrid<T>::find(value_type value) const
{
    return this->find_impl(value) - data_.begin();
}

//---------------------------------------------------------------------------//
/*!
 * Find the index (must be in bounds), and whether the value is on an edge
 *
 * This method behaves the same as the "find" method, accept it also
 * returns a bool denoting whether or not the value is coincident with the
 * lower edge of the bin.
 */
template<class T>
CELER_FUNCTION typename NonuniformGrid<T>::result_type
NonuniformGrid<T>::find_edge(value_type value) const
{
    auto iter = this->find_impl(value);
    return {iter - data_.begin(), *iter == value};
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return iterator for bin corresponding to value (must be in bounds)
 */
template<class T>
CELER_FUNCTION typename NonuniformGrid<T>::Data::iterator
NonuniformGrid<T>::find_impl(value_type value) const
{
    CELER_EXPECT(value >= this->front() && value < this->back());

    auto iter = celeritas::lower_bound(data_.begin(), data_.end(), value);
    CELER_ASSERT(iter != data_.end());

    if (value != *iter)
    {
        // Exactly on end grid point, or not on a grid point at all: move
        // to previous bin
        --iter;
    }
    return iter;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
