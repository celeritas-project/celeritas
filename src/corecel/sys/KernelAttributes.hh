//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelAttributes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "Device.hh"

#if CELER_DEVICE_SOURCE
#    include "corecel/DeviceRuntimeApi.hh"
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
 *
 * The special value of zero threads per block causes the kernel attributes to
 * default to the *compile-time maximum* number of threads per block as
 * specified by launch bounds.
 */
template<class F>
KernelAttributes
make_kernel_attributes(F* func, unsigned int threads_per_block = 0)
{
    KernelAttributes result;
#ifdef CELER_DEVICE_SOURCE
    // Get function attributes
    {
        CELER_DEVICE_API_SYMBOL(FuncAttributes) attr;
        CELER_DEVICE_API_CALL(
            FuncGetAttributes(&attr, reinterpret_cast<void const*>(func)));
        result.num_regs = attr.numRegs;
        result.const_mem = attr.constSizeBytes;
        result.local_mem = attr.localSizeBytes;
        result.max_threads_per_block = attr.maxThreadsPerBlock;
    }

    if (threads_per_block == 0)
    {
        // Use the maximum number of threads instead of having smaller blocks
        threads_per_block = result.max_threads_per_block;
    }

    // Get maximum number of active blocks per SM
    std::size_t dynamic_smem_size = 0;
    int num_blocks = 0;
    CELER_DEVICE_API_CALL(OccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, func, threads_per_block, dynamic_smem_size));
    result.max_blocks_per_cu = num_blocks;

    // Calculate occupancy statistics used for launch bounds
    // (threads / block) * (blocks / cu) * (cu / eu) * (warp / thread)
    Device const& d = celeritas::device();

    result.max_warps_per_eu = (threads_per_block * num_blocks)
                              / (d.eu_per_cu() * d.threads_per_warp());
    result.occupancy = static_cast<double>(num_blocks * threads_per_block)
                       / static_cast<double>(d.max_threads_per_cu());

    // Get size limits
    if constexpr (CELERITAS_USE_CUDA)
    {
        // Stack size limit is CUDA-only
        CELER_DEVICE_API_CALL(DeviceGetLimit(
            &result.stack_size, CELER_DEVICE_API_SYMBOL(LimitStackSize)));
        // HIP throws 'limit is not supported on this architecture'
        CELER_DEVICE_API_CALL(
            DeviceGetLimit(&result.print_buffer_size,
                           CELER_DEVICE_API_SYMBOL(LimitPrintfFifoSize)));
    }
    CELER_DEVICE_API_CALL(DeviceGetLimit(
        &result.heap_size, CELER_DEVICE_API_SYMBOL(LimitMallocHeapSize)));
#else
    CELER_DISCARD(func);
    CELER_ASSERT_UNREACHABLE();
#endif
    result.threads_per_block = threads_per_block;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
