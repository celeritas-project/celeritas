//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PoolBuilder.i.hh
//---------------------------------------------------------------------------//

#include <limits>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M>
void PoolBuilder<T, M>::reserve(size_type count)
{
    CELER_EXPECT(count <= max_pool_size());
    this->storage().reserve(count);
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given elements at the end of the allocation.
 */
template<class T, MemSpace M>
template<class InputIterator>
auto PoolBuilder<T, M>::insert_back(InputIterator first, InputIterator last)
    -> PoolRangeT
{
    CELER_EXPECT(std::distance(first, last) + this->storage().size()
                 <= this->max_pool_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    auto start = this->size();
    this->storage().insert(this->storage().end(), first, last);
    return {start, this->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, MemSpace M>
auto PoolBuilder<T, M>::insert_back(std::initializer_list<T> init)
    -> PoolRangeT
{
    return this->insert_back(init.begin(), init.end());
}

//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M>
auto PoolBuilder<T, M>::push_back(T el) -> PoolSize
{
    CELER_EXPECT(this->storage().size() + 1 <= this->max_pool_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");

    auto start = this->size();
    this->storage().push_back(el);
    return start;
}

//---------------------------------------------------------------------------//
// DEVICE POOL BUILDDER
//---------------------------------------------------------------------------//
/*!
 * Increase the size to the given number of elements.
 *
 * \todo Rethink whether to add resizing to DeviceVector, since this
 * construction is super awkward.
 */
template<class T, MemSpace M>
void PoolBuilder<T, M>::resize(size_type size)
{
    CELER_EXPECT(size >= this->size());
    CELER_EXPECT(this->storage().empty() || size <= this->storage().capacity());
    if (this->storage().empty())
    {
        this->storage() = StorageT(size);
    }
    else
    {
        this->storage().resize(size);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
