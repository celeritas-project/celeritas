//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CopierImpl.cc
//---------------------------------------------------------------------------//
#include "CopierImpl.hh"

#include <cstring>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
#if CELER_USE_DEVICE
inline auto to_memcpy_kind(MemSpace src, MemSpace dst)
{
    if (src != MemSpace::device && dst == MemSpace::device)
        return CELER_DEVICE_API_SYMBOL(MemcpyHostToDevice);
    else if (src == MemSpace::device && dst != MemSpace::device)
        return CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost);
    else if (src == MemSpace::device && dst == MemSpace::device)
        return CELER_DEVICE_API_SYMBOL(MemcpyDeviceToDevice);
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on a lazily obtained native stream.
 *
 * The stream is only requested if a device is involved in the copy, since
 * host-only copies may use stream IDs for which no device stream exists.
 */
template<class F>
void copy_bytes_async(MemSpace dstmem,
                      void* dst,
                      MemSpace srcmem,
                      void const* src,
                      std::size_t count,
                      F&& get_stream)
{
    if (srcmem != MemSpace::device && dstmem != MemSpace::device)
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DISCARD(get_stream);
    CELER_DEVICE_API_CALL(MemcpyAsync(
        dst, src, count, to_memcpy_kind(srcmem, dstmem), get_stream()));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Perform a memcpy on the data.
 */
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count)
{
    if (srcmem != MemSpace::device && dstmem != MemSpace::device)
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DEVICE_API_CALL(
        Memcpy(dst, src, count, to_memcpy_kind(srcmem, dstmem)));
}

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on the stream with the given ID.
 */
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count,
                     StreamId stream)
{
    copy_bytes_async(dstmem, dst, srcmem, src, count, [stream] {
        return stream ? celeritas::device().stream(stream).get() : nullptr;
    });
}

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on the given stream.
 */
void copy_bytes_impl(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count,
                     Stream const& stream)
{
    copy_bytes_async(
        dstmem, dst, srcmem, src, count, [&stream] { return stream.get(); });
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
