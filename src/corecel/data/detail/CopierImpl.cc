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
//! Whether neither side of a copy is in device memory
inline bool is_host_copy(MemSpace dstmem, MemSpace srcmem)
{
    return dstmem != MemSpace::device && srcmem != MemSpace::device;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Perform a memcpy on the data.
 */
void copy_bytes_sync(MemSpace dstmem,
                     void* dst,
                     MemSpace srcmem,
                     void const* src,
                     std::size_t count)
{
    if (is_host_copy(dstmem, srcmem))
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DEVICE_API_CALL(
        Memcpy(dst, src, count, to_memcpy_kind(srcmem, dstmem)));
}

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on the given stream.
 *
 * Copies between host memory spaces are performed immediately.
 */
void copy_bytes_async(MemSpace dstmem,
                      void* dst,
                      MemSpace srcmem,
                      void const* src,
                      std::size_t count,
                      Stream const& stream)
{
    if (is_host_copy(dstmem, srcmem))
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DISCARD(stream);
    CELER_DEVICE_API_CALL(MemcpyAsync(
        dst, src, count, to_memcpy_kind(srcmem, dstmem), stream.get()));
}

//---------------------------------------------------------------------------//
/*!
 * Perform an asynchronous memcpy on the stream with the given ID.
 *
 * Copies between host memory spaces are performed immediately without
 * accessing the device stream, which doesn't exist in CPU-only runs.
 */
void copy_bytes_async(MemSpace dstmem,
                      void* dst,
                      MemSpace srcmem,
                      void const* src,
                      std::size_t count,
                      StreamId stream)
{
    CELER_EXPECT(stream);
    if (is_host_copy(dstmem, srcmem))
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DISCARD(stream);
    CELER_DEVICE_API_CALL(
        MemcpyAsync(dst,
                    src,
                    count,
                    to_memcpy_kind(srcmem, dstmem),
                    celeritas::device().stream(stream).get()));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
