//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Copier.cc
//---------------------------------------------------------------------------//
#include "Copier.hh"

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif
#include <cstring>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Perform a memcpy on the data.
 */
void copy_bytes(MemSpace    dstmem,
                void*       dst,
                MemSpace    srcmem,
                const void* src,
                std::size_t count)
{
    if (srcmem == MemSpace::host && dstmem == MemSpace::host)
    {
        std::memcpy(dst, src, count);
        return;
    }

#if CELERITAS_USE_CUDA
    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (srcmem == MemSpace::host && dstmem == MemSpace::device)
        kind = cudaMemcpyHostToDevice;
    else if (srcmem == MemSpace::device && dstmem == MemSpace::host)
        kind = cudaMemcpyDeviceToHost;
    else if (srcmem == MemSpace::device && dstmem == MemSpace::device)
        kind = cudaMemcpyDeviceToDevice;
    else
        CELER_ASSERT_UNREACHABLE();
#endif
    CELER_CUDA_CALL(cudaMemcpy(dst, src, count, kind));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
