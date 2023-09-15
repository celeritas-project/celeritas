//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Copy spans of data.
 *
 * The destination (which can be a reusable buffer) is the constructor
 * argument, and the source of the data to copy is the function argument.
 *
 * Example of copying data from device to host:
 * \code
    Copier<int, MemSpace::host> copy_to_host{host_ints};
    copy_to_host(MemSpace::device, device_ints);
 * \endcode
 */
template<class T, MemSpace M>
struct Copier
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "Data is not trivially copyable");

    Span<T> dst;
    static constexpr auto dstmem = M;

    inline void operator()(MemSpace srcmem, Span<T const> src) const;
    inline void
    operator()(MemSpace srcmem, Span<T const> src, StreamId stream) const;
};

//---------------------------------------------------------------------------//
// Copy bytes between two memory spaces
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count);

// Asynchronously copy bytes between two memory spaces
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count,
                StreamId stream);

//---------------------------------------------------------------------------//
/*!
 * Copy data from the given source and memory space.
 */
template<class T, MemSpace M>
void Copier<T, M>::operator()(MemSpace srcmem, Span<T const> src) const
{
    CELER_EXPECT(src.size() == dst.size());
    copy_bytes(dstmem, dst.data(), srcmem, src.data(), src.size() * sizeof(T));
}

/*!
 * Asynchronously copy data from the given source and memory space on the given
 * stream.
 */
template<class T, MemSpace M>
void Copier<T, M>::operator()(MemSpace srcmem,
                              Span<T const> src,
                              StreamId stream) const
{
    CELER_EXPECT(src.size() == dst.size());
    copy_bytes(
        dstmem, dst.data(), srcmem, src.data(), src.size() * sizeof(T), stream);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
