//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PieImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../PieTypes.hh"
#include "base/Span.hh"

#ifndef __CUDA_ARCH__
#    include <limits>
#    include <vector>
#    include "base/Assert.hh"
#    include "base/DeviceVector.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Indexing type for pies.
using pie_size_type = unsigned int;

//---------------------------------------------------------------------------//
template<class T, Ownership W>
struct PieTraits
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
struct PieTraits<T, Ownership::const_reference>
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
//! Memspace-dependent type traits for a pie
template<class T, Ownership W, MemSpace M>
struct PieStorage
{
    using type = typename PieTraits<T, W>::SpanT;
    type data;
};

template<class T>
struct PieStorage<T, Ownership::value, MemSpace::host>;
template<class T>
struct PieStorage<T, Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Assignment semantics for a pie
template<Ownership W, MemSpace M>
struct PieAssigner
{
    template<class T, Ownership W2, MemSpace M2>
    PieStorage<T, W, M> operator()(const PieStorage<T, W2, M2>& source)
    {
        static_assert(W != Ownership::reference || W2 == W,
                      "Can't create a reference from a const reference");
        static_assert(M == M2, "Pie assignment from a different memory space");
        return {{source.data.data(), source.data.size()}};
    }

    template<class T, Ownership W2, MemSpace M2>
    PieStorage<T, W, M> operator()(PieStorage<T, W2, M2>& source)
    {
        static_assert(M == M2, "Pie assignment from a different memory space");
        static_assert(
            !(W == Ownership::reference && W2 == Ownership::const_reference),
            "Can't create a reference from a const reference");
        return {{source.data.data(), source.data.size()}};
    }
};

template<>
struct PieAssigner<Ownership::value, MemSpace::host>;
template<>
struct PieAssigner<Ownership::value, MemSpace::device>;

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
//! Storage implementation for managed host data
template<class T>
struct PieStorage<T, Ownership::value, MemSpace::host>
{
    using type = std::vector<T>;
    type data;
};

//! Storage implementation for managed device data
template<class T>
struct PieStorage<T, Ownership::value, MemSpace::device>
{
    using type = DeviceVector<T>;
    type data;
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to host memory
template<>
struct PieAssigner<Ownership::value, MemSpace::host>
{
    template<class T, Ownership W2, MemSpace M2>
    PieStorage<T, Ownership::value, MemSpace::host>
    operator()(const PieStorage<T, W2, M2>& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign host values from host data");
        return {{source.data.data(), source.data.data() + source.data.size()}};
    }
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to device memory
template<>
struct PieAssigner<Ownership::value, MemSpace::device>
{
    template<class T, Ownership W2, MemSpace M2>
    PieStorage<T, Ownership::value, MemSpace::device>
    operator()(const PieStorage<T, W2, M2>& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign by value from host to device");

        // Always-on once-per-copy check that the number of elements in the
        // vector doesn't exceed the range. This is in case of a production run
        // with lots of cross sections, when the incremental PieBuilder checks
        // are disabled. See comment in PieSlice class documentation.
        constexpr std::size_t max_pie_size
            = std::numeric_limits<pie_size_type>::max();
        CELER_VALIDATE(source.data.size() <= max_pie_size,
                       "Pie is too large for PieSlice: "
                           << source.data.size() << " > " << max_pie_size);

        PieStorage<T, Ownership::value, MemSpace::device> result{
            DeviceVector<T>(source.data.size())};
        result.data.copy_to_device({source.data.data(), source.data.size()});
        return result;
    }
};
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
