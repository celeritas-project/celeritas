//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionBuilder.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
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
auto CollectionBuilder<T, M, I>::push_back(const T& el) -> ItemIdT
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
// DEVICE COLLECTION BUILDER
//---------------------------------------------------------------------------//
/*!
 * Increase the size to the given number of elements.
 */
template<class T, MemSpace M, class I>
void CollectionBuilder<T, M, I>::resize(size_type size)
{
    CELER_EXPECT(this->storage().empty());
    this->storage() = StorageT(size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
