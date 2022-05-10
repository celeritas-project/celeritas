//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelParamCalculator.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"

#include "Device.hh"
#include "KernelDiagnostics.hh"
#include "ThreadId.hh"

/*!
 * \def CELER_LAUNCH_KERNEL
 *
 * Create a kernel param calculator with the given kernel, assuming the
 * function itself has a \c _kernel suffix, and launch with the given
 * block/thread sizes and arguments list.
 */
#define CELER_LAUNCH_KERNEL(NAME, BLOCK_SIZE, THREADS, ...)                  \
    do                                                                       \
    {                                                                        \
        static const ::celeritas::KernelParamCalculator calc_launch_params_( \
            NAME##_kernel, #NAME, BLOCK_SIZE);                               \
        auto grid_ = calc_launch_params_(THREADS);                           \
                                                                             \
        CELER_LAUNCH_KERNEL_IMPL(NAME##_kernel,                              \
                                 grid_.blocks_per_grid,                      \
                                 grid_.threads_per_block,                    \
                                 0,                                          \
                                 0,                                          \
                                 __VA_ARGS__);                               \
        CELER_DEVICE_CHECK_ERROR();                                          \
    } while (0)

#if CELERITAS_USE_CUDA
#    define CELER_LAUNCH_KERNEL_IMPL(KERNEL, GRID, BLOCK, SHARED, STREAM, ...) \
        KERNEL<<<GRID, BLOCK, SHARED, STREAM>>>(__VA_ARGS__)
#elif CELERITAS_USE_HIP
#    define CELER_LAUNCH_KERNEL_IMPL(KERNEL, GRID, BLOCK, SHARED, STREAM, ...) \
        hipLaunchKernelGGL(KERNEL, GRID, BLOCK, SHARED, STREAM, __VA_ARGS__)
#else
#    define CELER_LAUNCH_KERNEL_IMPL(KERNEL, GRID, BLOCK, SHARED, STREAM, ...) \
        CELER_NOT_CONFIGURED("CUDA or HIP");                                   \
        (void)sizeof(GRID);                                                    \
        (void)sizeof(KERNEL);                                                  \
        (void)sizeof(__VA_ARGS__);
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Kernel management helper functions.
 *
 * We assume that all our kernel launches use 1-D thread indexing to make
 * things easy. The \c dim_type alias should be the same size as the type of a
 * single \c dim3 member (x/y/z).
 *
 * Constructing the param calculator registers kernel diagnostics.
 *
 * \code
    static KernelParamCalculator calc_params(my_kernel, "my");
    auto params = calc_params(states.size());
    my_kernel<<<params.blocks_per_grid,
 params.threads_per_block>>>(kernel_args...); \endcode
 */
class KernelParamCalculator
{
  public:
    //!@{
    //! Type aliases
    using dim_type = unsigned int;
    using KernelId = OpaqueId<struct Kernel>;
    //!@}

    //! Parameters needed for a CUDA lauch call
    struct LaunchParams
    {
        dim3 blocks_per_grid;   //!< Number of blocks for kernel grid
        dim3 threads_per_block; //!< Number of threads per block
    };

  public:
    // Get the thread ID for a kernel initialized with this class
    inline CELER_FUNCTION static ThreadId thread_id();

    //// CLASS INTERFACE ////

    // Construct with the default block size
    template<class F>
    inline KernelParamCalculator(F* kernel_func_ptr, const char* name);

    // Construct with an explicit number of threads per block
    template<class F>
    inline KernelParamCalculator(F*          kernel_func_ptr,
                                 const char* name,
                                 dim_type    threads_per_block);

    // Get launch parameters
    LaunchParams operator()(size_type min_num_threads) const;

  private:
    //! Threads per block
    dim_type block_size_;
    //! Unique run-dependent ID for the associated kernel
    KernelId id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the linear thread ID.
 */
CELER_FUNCTION auto KernelParamCalculator::thread_id() -> ThreadId
{
#ifdef CELER_DEVICE_COMPILE
    return ThreadId{blockIdx.x * blockDim.x + threadIdx.x};
#else
    // blockIdx/threadIdx not available: shouldn't be called by host code
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Construct for the given global kernel F.
 */
template<class F>
KernelParamCalculator::KernelParamCalculator(F*          kernel_func_ptr,
                                             const char* name)
    : KernelParamCalculator(
        kernel_func_ptr, name, celeritas::device().default_block_size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct for the given global kernel F.
 */
template<class F>
KernelParamCalculator::KernelParamCalculator(F*          kernel_func_ptr,
                                             const char* name,
                                             dim_type    threads_per_block)
    : block_size_(threads_per_block)
{
    CELER_EXPECT(threads_per_block % celeritas::device().threads_per_warp()
                 == 0);
    id_ = celeritas::kernel_diagnostics().insert<F>(
        kernel_func_ptr, name, threads_per_block);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
