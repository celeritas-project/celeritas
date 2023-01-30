//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <limits>

#include "celeritas_config.h"

#include "Collection.hh"
#include "detail/FillInvalid.hh"

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
    ItemRange<int> insertion
        = cb.insert_back(local_ints.begin(), local_ints.end());
    cb.push_back(123);
   \endcode

 * The CollectionBuilder can also be used to resize device-value collections
 * without having to allocate a host version and copy to device. (This is
 * useful for state allocations.) When resizing values with debugging
 * assertions enabled on host memory, it will assign garbage values to aid in
 * reproducible debugging.)
 */
template<class T, MemSpace M, class I>
class CollectionBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using CollectionT = Collection<T, Ownership::value, M, I>;
    using value_type = T;
    using size_type = typename CollectionT::size_type;
    using ItemIdT = typename CollectionT::ItemIdT;
    using ItemRangeT = typename CollectionT::ItemRangeT;
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
    inline ItemIdT push_back(value_type const& element);
    inline ItemIdT push_back(value_type&& element);

    //! Number of elements in the collection
    size_type size() const { return col_.size(); }

  private:
    CollectionT& col_;

    using StorageT = typename CollectionT::StorageT;
    StorageT& storage() { return col_.storage(); }
    StorageT const& storage() const { return col_.storage(); }

    //! Maximum elements in a Collection, in underlying storage size
    static constexpr size_type max_size()
    {
        return std::numeric_limits<size_type>::max();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M, class I>
void CollectionBuilder<T, M, I>::reserve(size_type count)
{
    CELER_EXPECT(count <= max_size());
    static_assert(M == MemSpace::host,
                  "Reserve currently works only for host memory");
    this->storage().reserve(count);
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given elements at the end of the allocation.
 */
template<class T, MemSpace M, class I>
template<class InputIterator>
auto CollectionBuilder<T, M, I>::insert_back(InputIterator first,
                                             InputIterator last) -> ItemRangeT
{
    CELER_EXPECT(std::distance(first, last) + this->storage().size()
                 <= this->max_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    auto start = ItemIdT{this->size()};
    this->storage().insert(this->storage().end(), first, last);
    return {start, ItemIdT{this->size()}};
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, MemSpace M, class I>
auto CollectionBuilder<T, M, I>::insert_back(std::initializer_list<T> init)
    -> ItemRangeT
{
    return this->insert_back(init.begin(), init.end());
}

//---------------------------------------------------------------------------//
/*!
 * Add a new element to the end of the allocation.
 */
template<class T, MemSpace M, class I>
auto CollectionBuilder<T, M, I>::push_back(T const& el) -> ItemIdT
{
    CELER_EXPECT(this->storage().size() + 1 <= this->max_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    size_type idx = this->size();
    this->storage().push_back(el);
    return ItemIdT{idx};
}

template<class T, MemSpace M, class I>
auto CollectionBuilder<T, M, I>::push_back(T&& el) -> ItemIdT
{
    CELER_EXPECT(this->storage().size() + 1 <= this->max_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    size_type idx = this->size();
    this->storage().push_back(std::move(el));
    return ItemIdT{idx};
}

//---------------------------------------------------------------------------//
/*!
 * Increase the size to the given number of elements.
 */
template<class T, MemSpace M, class I>
void CollectionBuilder<T, M, I>::resize(size_type size)
{
    CELER_EXPECT(this->storage().empty());
    this->storage() = StorageT(size);
    if (CELERITAS_DEBUG && M == MemSpace::host)
    {
        // Fill with invalid values to help with debugging on host
        detail::fill_invalid(&col_);
    }
}

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
/*!
 * Helper function for resizing a single collection.
 *
 * This is useful for analogy to the resize method defined for states.
 */
template<class T, MemSpace M, class I>
void resize(Collection<T, Ownership::value, M, I>* collection,
            typename I::size_type size)
{
    CELER_EXPECT(collection);
    CollectionBuilder<T, M, I>(collection).resize(size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
