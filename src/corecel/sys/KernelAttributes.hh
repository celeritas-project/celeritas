//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelAttributes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "Device.hh"

#if CELER_DEVICE_SOURCE
#    include "corecel/device_runtime_api.h"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Immutable attributes of a kernel function.
 *
 * This is an analog to \c cudaFuncAttributes with some additional helpful
 * information. Some quantities are device-specific.
 */
struct KernelAttributes
{
    unsigned int threads_per_block{0};

    int num_regs{0};  //!< Number of 32-bit registers per thread
    std::size_t const_mem{0};  //!< Amount of constant memory (per thread) [b]
    std::size_t local_mem{0};  //!< Amount of local memory (per thread) [b]

    unsigned int max_threads_per_block{0};  //!< Max allowed threads per block
    unsigned int max_blocks_per_cu{0};  //!< Occupancy (compute unit)

    // Derivative but useful occupancy information
    unsigned int max_warps_per_eu{0};  //!< Occupancy (execution unit)
    double occupancy{0};  //!< Fractional occupancy (CU)

    // Resource limits at first call
    std::size_t stack_size{0};  //!< CUDA Dynamic per-thread stack limit [b]
    std::size_t heap_size{0};  //!< Dynamic malloc heap size [b]
    std::size_t print_buffer_size{0};  //!< FIFO buffer size for printf [b]
};

//---------------------------------------------------------------------------//
/*!
 * Build kernel attributes from a __global__ kernel function.
 *
 * This can only be called from CUDA/HIP code. It assumes that the block size
 * is constant across the execution of the program and that the kernel is only
 * called by the device that's active at this time.
 */
template<class F>
KernelAttributes make_kernel_attributes(F* func, unsigned int threads_per_block)
{
    KernelAttributes result;
    result.threads_per_block = threads_per_block;
#ifdef CELER_DEVICE_SOURCE
    // Get function attributes
    {
        CELER_DEVICE_PREFIX(FuncAttributes) attr;
        CELER_DEVICE_CALL_PREFIX(
            FuncGetAttributes(&attr, reinterpret_cast<void const*>(func)));
        result.num_regs = attr.numRegs;
        result.const_mem = attr.constSizeBytes;
        result.local_mem = attr.localSizeBytes;
        result.max_threads_per_block = attr.maxThreadsPerBlock;
    }

    // Get maximum number of active blocks per SM
    std::size_t dynamic_smem_size = 0;
    int num_blocks = 0;
    CELER_DEVICE_CALL_PREFIX(OccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, func, result.threads_per_block, dynamic_smem_size));
    result.max_blocks_per_cu = num_blocks;

    // Calculate occupancy statistics used for launch bounds
    // (threads / block) * (blocks / cu) * (cu / eu) * (warp / thread)
    Device const& d = celeritas::device();

    result.max_warps_per_eu = (threads_per_block * num_blocks)
                              / (d.eu_per_cu() * d.threads_per_warp());
    result.occupancy
        = static_cast<double>(num_blocks * result.threads_per_block)
          / static_cast<double>(d.max_threads_per_cu());

    // Get size limits
    if constexpr (CELERITAS_USE_CUDA)
    {
        // Stack size limit is CUDA-only
        CELER_CUDA_CALL(
            cudaDeviceGetLimit(&result.stack_size, cudaLimitStackSize));
    }
    CELER_DEVICE_CALL_PREFIX(DeviceGetLimit(
        &result.heap_size, CELER_DEVICE_PREFIX(LimitMallocHeapSize)));
    CELER_DEVICE_CALL_PREFIX(DeviceGetLimit(
        &result.print_buffer_size, CELER_DEVICE_PREFIX(LimitPrintfFifoSize)));

#else
    (void)sizeof(func);
    CELER_ASSERT_UNREACHABLE();
#endif
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
