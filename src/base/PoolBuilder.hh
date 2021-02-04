//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PoolBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <limits>
#include "Pool.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing Pools.
 *
 * This is intended for use with host data but can also be used to resize
 * device pools.
 *
 * \code
    auto pb = make_pool_builder(myintpool.host);
    pb.reserve(100);
    PoolRange<int> insertion = pb.extend(local_ints.begin(), local_ints.end());
    pb.push_back(
   \endcode
 */
template<class T, MemSpace M>
class PoolBuilder
{
  public:
    //!@{
    //! Type aliases
    using value_type = T;
    using PoolT      = Pool<T, Ownership::value, M>;
    using PoolRangeT = PoolRange<T>;
    using PoolSize   = typename PoolRangeT::size_type;
    //!@}

  public:
    //! Construct from a pool
    explicit PoolBuilder(PoolT& pool) : pool_(pool) {}

    // Increase size to this capacity
    inline void resize(size_type count);

    // Reserve space
    inline void reserve(size_type count);

    // Append a series of elements, returning the range inserted
    template<class InputIterator>
    inline PoolRangeT insert_back(InputIterator first, InputIterator last);

    // Append a series of elements from an initializer list
    inline PoolRangeT insert_back(std::initializer_list<T> init);

    // Append a single element
    inline PoolSize push_back(value_type element);

    //! Number of elements in the pool
    PoolSize size() const { return pool_.size(); }

  private:
    PoolT& pool_;

    using StorageT = typename PoolT::StorageT;
    StorageT&       storage() { return pool_.storage(); }
    const StorageT& storage() const { return pool_.storage(); }

    //! Maximum elements in a Pool, in native std::size_t
    static constexpr size_type max_pool_size()
    {
        return std::numeric_limits<PoolSize>::max();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing pool builders.
 *
 * (Will not be needed under C++20's new constructor lookups).
 */
template<class T, MemSpace M>
PoolBuilder<T, M> make_pool_builder(Pool<T, Ownership::value, M>& pool)
{
    return PoolBuilder<T, M>(pool);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PoolBuilder.i.hh"
