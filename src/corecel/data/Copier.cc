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
template<bool async>
inline void copy_bytes_impl(MemSpace dstmem,
                            void* dst,
                            MemSpace srcmem,
                            void const* src,
                            std::size_t count,
                            [[maybe_unused]] StreamId stream)
{
    if (srcmem == MemSpace::host && dstmem == MemSpace::host)
    {
        std::memcpy(dst, src, count);
        return;
    }

#if CELER_USE_DEVICE
    CELER_DEVICE_PREFIX(MemcpyKind) kind = CELER_DEVICE_PREFIX(MemcpyDefault);
    if (srcmem == MemSpace::host && dstmem == MemSpace::device)
        kind = CELER_DEVICE_PREFIX(MemcpyHostToDevice);
    else if (srcmem == MemSpace::device && dstmem == MemSpace::host)
        kind = CELER_DEVICE_PREFIX(MemcpyDeviceToHost);
    else if (srcmem == MemSpace::device && dstmem == MemSpace::device)
        kind = CELER_DEVICE_PREFIX(MemcpyDeviceToDevice);
    else
        CELER_ASSERT_UNREACHABLE();
#endif
    if constexpr (async)
    {
        CELER_DEVICE_CALL_PREFIX(MemcpyAsync(
            dst, src, count, kind, celeritas::device().stream(stream).get()));
    }
    else
    {
        CELER_DEVICE_CALL_PREFIX(Memcpy(dst, src, count, kind));
    }
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
    copy_bytes_impl<false>(dstmem, dst, srcmem, src, count, StreamId{0});
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
    copy_bytes_impl<true>(dstmem, dst, srcmem, src, count, stream);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
