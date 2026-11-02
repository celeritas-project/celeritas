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
#include "corecel/sys/Device.hh"
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
    Copier copy_to_host{ObserverPtr<int, MemSpace::host>{host_ints.data()},
                        host_ints.size(),
                        StreamId{}};
    copy_to_host(MemSpace::device, device_ints);
 * \endcode
 */
template<class T, MemSpace M>
class Copier
{
    static_assert(is_trivially_copyable_v<T>, "Data is not trivially copyable");

  public:
    //! Construct with the destination, size, and stream for async copy
    Copier(ObserverPtr<T, M> dst, std::size_t size, StreamId stream)
        : dst_{dst}, size_{size}, stream_{stream}
    {
    }

    //! Construct with a single dst element
    Copier(ObserverPtr<T, M> dst, StreamId stream)
        : dst_{dst}, size_{1}, stream_{stream}
    {
    }

    //! Construct from a destination span and stream
    Copier(Span<T> dst, StreamId stream)
        : Copier{ObserverPtr<T, M>{dst.data()}, dst.size(), stream}
    {
    }

    // Load from a span of data
    inline void operator()(MemSpace srcmem, Span<T const> src) const;

    // Load from a single element
    template<MemSpace M2>
    inline void operator()(ObserverPtr<T const, M2>) const;

    template<MemSpace M2>
    inline void operator()(ObserverPtr<T, M2> src) const
    {
        return (*this)(ObserverPtr<T const, M2>{src});
    }

  private:
    ObserverPtr<T, M> dst_;
    std::size_t size_;
    StreamId stream_;
    static constexpr auto dstmem = M;
};

//---------------------------------------------------------------------------//
//! Deduce the element type and memory space from the destination pointer.
template<class T, MemSpace M>
Copier(ObserverPtr<T, M>, std::size_t, StreamId) -> Copier<T, M>;

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//

/*!
 * Copy a single value from the given memory space to the host.
 *
 * If the source is on the device, the copy is enqueued on the given stream
 * and the stream is synchronized before returning.
 */
template<class T, MemSpace M>
inline std::remove_const_t<T> load_sync(ObserverPtr<T, M> src,
                                        StreamId stream = {})
{
    if constexpr (!std::is_const_v<T>)
    {
        return load_sync(ObserverPtr<T const, M>{src}, stream);
    }
    else
    {
        using value_type = std::remove_const_t<T>;
        static_assert(is_trivially_copyable_v<value_type>,
                      "Data is not trivially copyable");

        CELER_EXPECT(src);
        if constexpr (M == MemSpace::host)
        {
            return *static_cast<value_type const*>(src);
        }
        else
        {
            value_type result{};
            Copier<value_type, MemSpace::host> copy_from{
                ObserverPtr<value_type, MemSpace::host>{&result}, 1, stream};
            copy_from(M, {static_cast<value_type const*>(src), 1});
            if (stream)
            {
                // Since result isn't pinned, the underlying copy call should
                // have synchronized, but sync explicitly to be clearer.
                device().stream(stream).sync();
            }
            return result;
        }
    }
}

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

/*!
 * Copy data from the given source and memory space.
 */
template<class T, MemSpace M>
void Copier<T, M>::operator()(MemSpace srcmem, Span<T const> src) const
{
    CELER_EXPECT(src.size() == size_);
    if (stream_)
    {
        copy_bytes(dstmem,
                   static_cast<T*>(dst_),
                   srcmem,
                   src.data(),
                   size_ * sizeof(T),
                   stream_);
    }
    else
    {
        copy_bytes(dstmem,
                   static_cast<T*>(dst_),
                   srcmem,
                   src.data(),
                   size_ * sizeof(T));
    }
}

/*!
 * Copy a single element of data from the given source and memory space.
 */
template<class T, MemSpace M>
template<MemSpace M2>
inline void Copier<T, M>::operator()(ObserverPtr<T const, M2> src) const
{
    CELER_EXPECT(src);
    return (*this)(M2, {static_cast<T const*>(src), 1});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
