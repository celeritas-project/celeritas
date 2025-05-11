//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DedupeCollectionBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cstddef>
#include <unordered_set>

#include "corecel/math/HashUtils.hh"

#include "Collection.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Build collections, returning the same ID for reused data spans.
 */
template<class T, class I = ItemId<T>>
class DedupeCollectionBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using CollectionT = Collection<T, Ownership::value, MemSpace::host, I>;
    using value_type = T;
    using size_type = typename CollectionT::size_type;
    using ItemIdT = typename CollectionT::ItemIdT;
    using ItemRangeT = typename CollectionT::ItemRangeT;
    //!@}

  public:
    // Construct from a collection
    explicit DedupeCollectionBuilder(CollectionT* collection);

    //// ACCESSORS ////

    //! Number of elements in the collection
    size_type size() const { return this->storage().size(); }

    //! Get the size as an ID type
    ItemIdT size_id() const { return ItemIdT{this->size()}; }

    //// MUTATORS ////

    // Reserve space and hash table space
    inline void reserve(std::size_t count);

    // Extend with a series of elements, returning the range inserted
    template<class InputIterator>
    inline ItemRangeT insert_back(InputIterator first, InputIterator last);

    // Extend with a series of elements from an initializer list
    inline ItemRangeT insert_back(std::initializer_list<value_type> init);

    // Append a single element
    inline ItemIdT push_back(value_type const& element);

  private:
    //// CLASSES ////

    using StorageT = typename CollectionT::StorageT;

    struct HashRange
    {
        StorageT* storage{nullptr};
        std::size_t operator()(ItemRangeT) const;
    };
    struct EqualRange
    {
        StorageT* storage{nullptr};
        bool operator()(ItemRangeT const&, ItemRangeT const&) const;
    };

    //// DATA ////

    CollectionT* col_;
    std::unordered_set<ItemRangeT, HashRange, EqualRange> ranges_;

    //// FUNCTIONS ////

    //!@{
    //! Access storage
    CELER_FORCEINLINE StorageT& storage() { return col_->storage(); }
    CELER_FORCEINLINE StorageT const& storage() const
    {
        return col_->storage();
    }
    //!@}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a collection.
 */
template<class T, class I>
DedupeCollectionBuilder<T, I>::DedupeCollectionBuilder(CollectionT* collection)
    : col_{collection}
    , ranges_{0, HashRange{&this->storage()}, EqualRange{&this->storage()}}
{
    CELER_EXPECT(col_);
}

//---------------------------------------------------------------------------//
/*!
 * Reserve space for elements.
 */
template<class T, class I>
void DedupeCollectionBuilder<T, I>::reserve(std::size_t count)
{
    this->storage().reserve(count);
    ranges_.reserve(count);
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, class I>
template<class InputIterator>
auto DedupeCollectionBuilder<T, I>::insert_back(InputIterator first,
                                                InputIterator last)
    -> ItemRangeT
{
    auto& s = this->storage();

    auto start = this->size_id();
    s.insert(s.end(), first, last);
    ItemRangeT result{start, this->size_id()};
    auto [iter, inserted] = ranges_.insert(result);
    if (!inserted)
    {
        // Roll back the change by erasing the last elements
        s.erase(s.begin() + start.unchecked_get(), s.end());
        // Return existing range
        return *iter;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, class I>
auto DedupeCollectionBuilder<T, I>::insert_back(std::initializer_list<T> init)
    -> ItemRangeT
{
    return this->insert_back(init.begin(), init.end());
}

//---------------------------------------------------------------------------//
/*!
 * Add a new element to the end of the allocation.
 */
template<class T, class I>
auto DedupeCollectionBuilder<T, I>::push_back(T const& el) -> ItemIdT
{
    auto result = this->size_id();
    this->storage().push_back(el);
    return result;
}

//---------------------------------------------------------------------------//
// NESTED CLASSES
//---------------------------------------------------------------------------//
/*!
 * Compute a hash of the data underlying a range.
 *
 * By default, this uses the std::hash specialization in \c HashUtils.
 */
template<class T, class I>
std::size_t
DedupeCollectionBuilder<T, I>::HashRange::operator()(ItemRangeT r) const
{
    CELER_EXPECT(*r.end() <= storage->size());

    // Hash data pointed to by range
    Span<T const> data{storage->data() + r.begin()->unchecked_get(), r.size()};
    return std::hash<decltype(data)>{}(data);
}

//---------------------------------------------------------------------------//
/*!
 * Compare as equal the data referenced by two ranges.
 */
template<class T, class I>
bool DedupeCollectionBuilder<T, I>::EqualRange::operator()(
    ItemRangeT const& a, ItemRangeT const& b) const
{
    CELER_EXPECT(*a.end() <= storage->size());
    CELER_EXPECT(*b.end() <= storage->size());

    auto values_equal = [&s = *storage](ItemIdT left, ItemIdT right) {
        return s[left.unchecked_get()] == s[right.unchecked_get()];
    };
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), values_equal);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
