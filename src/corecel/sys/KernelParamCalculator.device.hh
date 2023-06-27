//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelParamCalculator.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <string_view>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

#include "Device.hh"
#include "KernelAttributes.hh"
#include "ThreadId.hh"  // IWYU pragma: export

/*!
 * \def CELER_LAUNCH_KERNEL
 *
 * Create a kernel param calculator with the given kernel, assuming the
 * function itself has a \c _kernel suffix, and launch with the given
 * block/thread sizes and arguments list.
 */
#define CELER_LAUNCH_KERNEL(NAME, BLOCK_SIZE, THREADS, STREAM, ...)          \
    do                                                                       \
    {                                                                        \
        static const ::celeritas::KernelParamCalculator calc_launch_params_( \
            #NAME, NAME##_kernel, BLOCK_SIZE);                               \
        auto grid_ = calc_launch_params_(THREADS);                           \
                                                                             \
        CELER_LAUNCH_KERNEL_IMPL(NAME##_kernel,                              \
                                 grid_.blocks_per_grid,                      \
                                 grid_.threads_per_block,                    \
                                 0,                                          \
                                 STREAM,                                     \
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
struct KernelProfiling;

//---------------------------------------------------------------------------//
/*!
 * Kernel management helper functions.
 *
 * We assume that all our kernel launches use 1-D thread indexing to make
 * things easy. The \c dim_type alias should be the same size as the type of a
 * single \c dim3 member (x/y/z).
 *
 * Constructing the param calculator registers kernel attributes with \c
 * kernel_registry as an implementation detail in the .cc file that hides
 * inclusion of that interface from CUDA code. If kernel diagnostic profiling
 * is enabled, the registry will return a pointer that this class uses to
 * increment thread launch counters over the lifetime of the program.
 *
 * \code
    static KernelParamCalculator calc_params("my", &my_kernel);
    auto params = calc_params(states.size());
    my_kernel<<<params.blocks_per_grid,
 params.threads_per_block>>>(kernel_args...);
 * \endcode
 */
class KernelParamCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using dim_type = unsigned int;
    //!@}

    //! Parameters needed for a CUDA lauch call
    struct LaunchParams
    {
        dim3 blocks_per_grid;  //!< Number of blocks for kernel grid
        dim3 threads_per_block;  //!< Number of threads per block
    };

  public:
    // Get the thread ID for a kernel initialized with this class
    inline CELER_FUNCTION static ThreadId thread_id();

    //// CLASS INTERFACE ////

    // Construct with the default block size
    template<class F>
    inline KernelParamCalculator(std::string_view name, F* kernel_func_ptr);

    // Construct with an explicit number of threads per block
    template<class F>
    inline KernelParamCalculator(std::string_view name,
                                 F* kernel_func_ptr,
                                 dim_type threads_per_block);

    // Get launch parameters
    inline LaunchParams operator()(size_type min_num_threads) const;

  private:
    //! Threads per block
    dim_type block_size_;
    //! Optional profiling data owned by the kernel registry
    KernelProfiling* profiling_{nullptr};

    //// HELPER FUNCTIONS ////

    void register_kernel(std::string_view name, KernelAttributes&& attributes);
    void log_launch(size_type min_num_threads) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the linear thread ID.
 */
CELER_FUNCTION auto KernelParamCalculator::thread_id() -> ThreadId
{
#if CELER_DEVICE_COMPILE
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
KernelParamCalculator::KernelParamCalculator(std::string_view name,
                                             F* kernel_func_ptr)
    : KernelParamCalculator(
        name, kernel_func_ptr, celeritas::device().default_block_size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct for the given global kernel F.
 *
 * This registers the kernel with \c celeritas::kernel_registry() and saves a
 * pointer to the profiling data if profiling is to be used.
 */
template<class F>
KernelParamCalculator::KernelParamCalculator(std::string_view name,
                                             F* kernel_func_ptr,
                                             dim_type threads_per_block)
    : block_size_(threads_per_block)
{
    CELER_EXPECT(threads_per_block <= static_cast<dim_type>(
                     celeritas::device().max_threads_per_block()));
    CELER_EXPECT(threads_per_block % celeritas::device().threads_per_warp()
                 == 0);

    this->register_kernel(
        name, make_kernel_attributes(kernel_func_ptr, threads_per_block));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate launch params given the number of threads.
 */
auto KernelParamCalculator::operator()(size_type min_num_threads) const
    -> LaunchParams
{
    CELER_EXPECT(min_num_threads > 0);

    // Update diagnostics for the kernel
    if (profiling_)
    {
        this->log_launch(min_num_threads);
    }

    // Ceiling integer division
    dim_type blocks_per_grid
        = celeritas::ceil_div<dim_type>(min_num_threads, this->block_size_);
    CELER_ASSERT(blocks_per_grid
                 < dim_type(celeritas::device().max_blocks_per_grid()));

    LaunchParams result;
    result.blocks_per_grid.x = blocks_per_grid;
    result.threads_per_block.x = this->block_size_;
    CELER_ENSURE(result.blocks_per_grid.x * result.threads_per_block.x
                 >= min_num_threads);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
