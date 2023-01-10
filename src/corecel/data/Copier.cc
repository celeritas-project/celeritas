//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.cc
//---------------------------------------------------------------------------//
#include "Copier.hh"

#include <cstring>

#include "corecel/device_runtime_api.h"
#include "corecel/Macros.hh"

namespace celeritas
{
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
    CELER_DEVICE_CALL_PREFIX(Memcpy(dst, src, count, kind));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
