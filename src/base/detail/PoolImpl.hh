//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PoolImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../PoolTypes.hh"
#include "base/Span.hh"

#ifndef __CUDA_ARCH__
#    include <vector>
#    include "base/DeviceVector.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, Ownership W>
struct PoolTraits
{
    using SpanT                = Span<T>;
    using SpanConstT           = Span<const T>;
    using pointer              = T*;
    using const_pointer        = const T*;
    using reference_type       = T&;
    using const_reference_type = const T&;
    using value_type           = T;
};

//---------------------------------------------------------------------------//
template<class T>
struct PoolTraits<T, Ownership::const_reference>
{
    using SpanT                = Span<const T>;
    using SpanConstT           = Span<const T>;
    using pointer              = const T*;
    using const_pointer        = const T*;
    using reference_type       = const T&;
    using const_reference_type = const T&;
    using value_type           = T;
};

//---------------------------------------------------------------------------//
//! Memspace-dependent type traits for a pool
template<class T, Ownership W, MemSpace M>
struct PoolStorage
{
    using type = typename PoolTraits<T, W>::SpanT;
    type data;
};

template<class T>
struct PoolStorage<T, Ownership::value, MemSpace::host>;
template<class T>
struct PoolStorage<T, Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Assignment semantics for a pool
template<Ownership W, MemSpace M>
struct PoolAssigner
{
    template<class T, Ownership W2, MemSpace M2>
    PoolStorage<T, W, M> operator()(const PoolStorage<T, W2, M2>& source)
    {
        static_assert(W != Ownership::reference || W2 == W,
                      "Can't create a reference from a const reference");
        static_assert(M == M2, "Pool assignment from a different memory space");
        return {{source.data.data(), source.data.size()}};
    }

    template<class T, Ownership W2, MemSpace M2>
    PoolStorage<T, W, M> operator()(PoolStorage<T, W2, M2>& source)
    {
        static_assert(M == M2, "Pool assignment from a different memory space");
        return {{source.data.data(), source.data.size()}};
    }
};

template<>
struct PoolAssigner<Ownership::value, MemSpace::device>;

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
//! Storage implementation for managed host data
template<class T>
struct PoolStorage<T, Ownership::value, MemSpace::host>
{
    using type = std::vector<T>;
    type data;
};

//! Storage implementation for managed device data
template<class T>
struct PoolStorage<T, Ownership::value, MemSpace::device>
{
    using type = DeviceVector<T>;
    type data;
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to device memory
template<>
struct PoolAssigner<Ownership::value, MemSpace::device>
{
    template<class T, Ownership W2, MemSpace M2>
    PoolStorage<T, Ownership::value, MemSpace::device>
    operator()(const PoolStorage<T, W2, M2>& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign by value from host to device");
        PoolStorage<T, Ownership::value, MemSpace::device> result{
            DeviceVector<T>(source.data.size())};
        result.data.copy_to_device({source.data.data(), source.data.size()});
        return result;
    }
};
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
