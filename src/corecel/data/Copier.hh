//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "ObserverPtr.hh"

#include "detail/CopierImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class Stream;

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
class Copier
{
    static_assert(is_trivially_copyable_v<T>, "Data is not trivially copyable");

  public:
    //! Construct with the destination and the class's memspace
    explicit Copier(Span<T> dst) : dst_{dst} {}

    //! Also construct with a stream ID to use for async copy
    Copier(Span<T> dst, StreamId stream) : dst_{dst}, stream_{stream} {}

    inline void operator()(MemSpace srcmem, Span<T const> src) const;

  private:
    Span<T> dst_;
    StreamId stream_;
    static constexpr auto dstmem = M;
};

//---------------------------------------------------------------------------//
/*!
 * Copy a single value from device to host.
 *
 * The source of the data to copy is the function argument.
 */
template<class T>
class ItemCopier
{
    static_assert(is_trivially_copyable_v<T>, "Data is not trivially copyable");

  public:
    //! Default constructor
    ItemCopier() = default;

    //! Also construct with a stream ID to use for async copy
    explicit ItemCopier(StreamId stream) : stream_{stream} {}

    inline T operator()(T const* src) const;

  private:
    StreamId stream_;
};

//---------------------------------------------------------------------------//
/*!
 * Copy bytes between two memory spaces.
 */
inline void copy_bytes(MemSpace dstmem,
                       void* dst,
                       MemSpace srcmem,
                       void const* src,
                       std::size_t count)
{
    detail::copy_bytes_sync(dstmem, dst, srcmem, src, count);
}

//---------------------------------------------------------------------------//
/*!
 * Asynchronously copy bytes between two memory spaces.
 */
inline void copy_bytes(MemSpace dstmem,
                       void* dst,
                       MemSpace srcmem,
                       void const* src,
                       std::size_t count,
                       StreamId stream)
{
    detail::copy_bytes_async(dstmem, dst, srcmem, src, count, stream);
}

//---------------------------------------------------------------------------//
/*!
 * Asynchronously copy bytes between two memory spaces on a stream.
 *
 * The memory spaces are part of the pointer types, so they can't be mixed up.
 * Copying to or from pageable host memory may be synchronous.
 */
template<MemSpace D, MemSpace S>
inline void copy_bytes(ObserverPtr<void, D> dst,
                       ObserverPtr<void const, S> src,
                       std::size_t count,
                       Stream const& stream)
{
    detail::copy_bytes_async(D,
                             static_cast<void*>(dst),
                             S,
                             static_cast<void const*>(src),
                             count,
                             stream);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from the given source and memory space.
 */
template<class T, MemSpace M>
void Copier<T, M>::operator()(MemSpace srcmem, Span<T const> src) const
{
    CELER_EXPECT(src.size() == dst_.size());
    if (stream_)
    {
        copy_bytes(dstmem,
                   dst_.data(),
                   srcmem,
                   src.data(),
                   src.size() * sizeof(T),
                   stream_);
    }
    else
    {
        copy_bytes(
            dstmem, dst_.data(), srcmem, src.data(), src.size() * sizeof(T));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Copy a value from device to host.
 */
template<class T>
T ItemCopier<T>::operator()(T const* src) const
{
    T dst;
    if (stream_)
    {
        copy_bytes(
            MemSpace::host, &dst, MemSpace::device, src, sizeof(T), stream_);
    }
    else
    {
        copy_bytes(MemSpace::host, &dst, MemSpace::device, src, sizeof(T));
    }
    return dst;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
