//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.cc
//---------------------------------------------------------------------------//
#include "Copier.hh"

#include <cstring>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Macros.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Perform a memcpy on the data.
 */
void copy_bytes(MemSpace dstmem,
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
 * Perform an asynchronous memcpy on the data.
 */
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count,
                StreamId stream)
{
    if (srcmem != MemSpace::device && dstmem != MemSpace::device)
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DISCARD(stream);
    CELER_DEVICE_API_CALL(MemcpyAsync(
        dst,
        src,
        count,
        to_memcpy_kind(srcmem, dstmem),
        stream ? celeritas::device().stream(stream).get() : nullptr));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
