//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.cc
//---------------------------------------------------------------------------//
#include "Copier.hh"

#include <cstring>

#include "corecel/device_runtime_api.h"
#include "corecel/Macros.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
namespace
{

#if CELER_USE_DEVICE
inline auto to_memcpy_kind(MemSpace src, MemSpace dst)
{
    if (src == MemSpace::host && dst == MemSpace::device)
        return CELER_DEVICE_PREFIX(MemcpyHostToDevice);
    else if (src == MemSpace::device && dst == MemSpace::host)
        return CELER_DEVICE_PREFIX(MemcpyDeviceToHost);
    else if (src == MemSpace::device && dst == MemSpace::device)
        return CELER_DEVICE_PREFIX(MemcpyDeviceToDevice);
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
    if (srcmem == MemSpace::host && dstmem == MemSpace::host)
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DEVICE_CALL_PREFIX(
        Memcpy(dst, src, count, to_memcpy_kind(srcmem, dstmem)));
}

/*!
 * Perform an asynchronous memcpy on the data.
 */
void copy_bytes(MemSpace dstmem,
                void* dst,
                MemSpace srcmem,
                void const* src,
                std::size_t count,
                CELER_UNUSED_UNLESS_DEVICE StreamId stream)
{
    if (srcmem == MemSpace::host && dstmem == MemSpace::host)
    {
        std::memcpy(dst, src, count);
        return;
    }
    CELER_DEVICE_CALL_PREFIX(
        MemcpyAsync(dst,
                    src,
                    count,
                    to_memcpy_kind(srcmem, dstmem),
                    celeritas::device().stream(stream).get()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
