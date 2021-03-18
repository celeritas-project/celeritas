//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <limits>
#include "Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing Collection objects.
 *
 * This is intended for use with host data but can also be used to resize
 * device collections. It's constructed with a reference to the host
 * collection, and it provides vector-like methods for extending it. The size
 * *cannot* be decreased because that would invalidate previously created \c
 * ItemRange items.
 *
 * \code
    auto cb = make_builder(&myintcol.host);
    cb.reserve(100);
    ItemRange<int> insertion = cb.extend(local_ints.begin(), local_ints.end());
    cb.push_back(123);
   \endcode

 * The CollectionBuilder can also be used to resize device-value collections
 * without having to allocate a host version and copy to device. (This is
 * useful for state allocations.)
 */
template<class T, MemSpace M, class I>
class CollectionBuilder
{
  public:
    //!@{
    //! Type aliases
    using CollectionT = Collection<T, Ownership::value, M, I>;
    using value_type  = T;
    using size_type   = typename CollectionT::size_type;
    using ItemIdT     = typename CollectionT::ItemIdT;
    using ItemRangeT  = typename CollectionT::ItemRangeT;
    //!@}

  public:
    //! Construct from a collection
    explicit CollectionBuilder(CollectionT* collection) : col_(*collection) {}

    // Increase size to this capacity
    inline void resize(size_type count);

    // Reserve space
    inline void reserve(size_type count);

    // Extend with a series of elements, returning the range inserted
    template<class InputIterator>
    inline ItemRangeT insert_back(InputIterator first, InputIterator last);

    // Extend with a series of elements from an initializer list
    inline ItemRangeT insert_back(std::initializer_list<value_type> init);

    // Append a single element
    inline ItemIdT push_back(value_type element);

    //! Number of elements in the collection
    size_type size() const { return col_.size(); }

  private:
    CollectionT& col_;

    using StorageT = typename CollectionT::StorageT;
    StorageT&       storage() { return col_.storage(); }
    const StorageT& storage() const { return col_.storage(); }

    //! Maximum elements in a Collection, in underlying storage size
    static constexpr size_type max_size()
    {
        return std::numeric_limits<size_type>::max();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper function for constructing collection builders.
 *
 * (Will not be needed under C++17's template argument deduction).
 */
template<class T, MemSpace M, class I>
CollectionBuilder<T, M, I>
make_builder(Collection<T, Ownership::value, M, I>* collection)
{
    CELER_EXPECT(collection);
    return CollectionBuilder<T, M, I>(collection);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "CollectionBuilder.i.hh"
