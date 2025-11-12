//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * UniformGrid. The grid is closed on the lower bound and open on the upper,
 * and the same treatment is used for coincident grid points: the higher of
 * the points will be chosen.
 */
template<class T>
class NonuniformGrid
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    using Storage
        = Collection<value_type, Ownership::const_reference, MemSpace::native>;
    using ItemRangeT = ItemRange<value_type>;
    using SpanConstT = typename Storage::SpanConstT;
    //!@}

  public:
    // Construct with storage
    inline CELER_FUNCTION
    NonuniformGrid(ItemRangeT const& values, Storage const& storage);

    //! Number of grid points
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        return offset_.size();
    }

    //! Minimum/first value
    CELER_FORCEINLINE_FUNCTION value_type front() const
    {
        return storage_[*offset_.begin()];
    }

    //! Maximum/last value
    CELER_FORCEINLINE_FUNCTION value_type back() const
    {
        return storage_[*(offset_.end() - 1)];
    }

    // Calculate the value at the given grid point
    inline CELER_FUNCTION value_type operator[](size_type i) const;

    // Find the index of the given value (*must* be in bounds)
    inline CELER_FUNCTION size_type find(value_type value) const;

    //! Low-level access to offsets for downstream utilities
    CELER_FORCEINLINE_FUNCTION ItemRangeT offset() const { return offset_; }

    // Construct a span referring to the grid points
    inline CELER_FUNCTION SpanConstT values() const;

  private:
    Storage const& storage_;
    ItemRangeT offset_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a range indexing into backend storage.
 */
template<class T>
CELER_FUNCTION NonuniformGrid<T>::NonuniformGrid(ItemRangeT const& values,
                                                 Storage const& storage)
    : storage_{storage}, offset_{values}
{
    CELER_EXPECT(offset_.size() >= 2);
    CELER_EXPECT(*offset_.end() <= storage.size());
    CELER_EXPECT(this->front() <= this->back());  // Approximation for "sorted"
}

//---------------------------------------------------------------------------//
/*!
 * Get the value at the given grid point.
 */
template<class T>
CELER_FUNCTION auto NonuniformGrid<T>::operator[](size_type i) const
    -> value_type
{
    CELER_EXPECT(i < offset_.size());
    return storage_[offset_[i]];
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

    auto iter = celeritas::upper_bound(
        offset_.begin() + 1,
        offset_.end() - 1,
        value,
        [&v = storage_](T lhs, ItemId<T> rhs_id) { return lhs < v[rhs_id]; });

    if (value < storage_[*iter])
    {
        // The start point belongs to the previous bin
        --iter;
    }

    return iter - offset_.begin();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a span referring to the grid points.
 */
template<class T>
CELER_FUNCTION auto NonuniformGrid<T>::values() const -> SpanConstT
{
    return storage_[offset_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
