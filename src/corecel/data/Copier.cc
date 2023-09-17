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

inline auto to_memcpy_kind(MemSpace src, MemSpace dst)
{
#if CELER_USE_DEVICE
    CELER_DEVICE_PREFIX(MemcpyKind) kind = CELER_DEVICE_PREFIX(MemcpyDefault);
    if (src == MemSpace::host && dst == MemSpace::device)
        kind = CELER_DEVICE_PREFIX(MemcpyHostToDevice);
    else if (src == MemSpace::device && dst == MemSpace::host)
        kind = CELER_DEVICE_PREFIX(MemcpyDeviceToHost);
    else if (src == MemSpace::device && dst == MemSpace::device)
        kind = CELER_DEVICE_PREFIX(MemcpyDeviceToDevice);
    else
        CELER_ASSERT_UNREACHABLE();
    return kind;
#endif
}

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
                StreamId stream)
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
