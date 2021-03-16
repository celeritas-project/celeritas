//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.cuda.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include "Assert.hh"
#include "Macros.hh"
#include "OpaqueId.hh"
#include "Types.hh"
#ifndef __CUDA_ARCH__
#    include "comm/Device.hh"
#    include "comm/KernelDiagnostics.hh"
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
    my_kernel<<<params.grid_size, params.block_size>>>(kernel_args...);
   \endcode
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
        dim3 grid_size;  //!< Number of blocks for kernel grid
        dim3 block_size; //!< Number of threads per block
    };

  public:
    // Get the thread ID for a kernel initialized with this class
    inline CELER_FUNCTION static ThreadId thread_id();

    //// CLASS INTERFACE ////

    // Construct with the default block size
    template<class F>
    KernelParamCalculator(F kernel_func, const char* name);

    // Construct with an explicit number of threads per block
    template<class F>
    KernelParamCalculator(F kernel_func, const char* name, dim_type block_size);

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
#ifdef __CUDA_ARCH__
    return ThreadId{blockIdx.x * blockDim.x + threadIdx.x};
#else
    // blockIdx/threadIdx not available: shouldn't be called by host code
    CELER_ASSERT_UNREACHABLE();
#endif
}

#ifndef __CUDA_ARCH__
// Hide host-side Device and KernelDiagnostsics from device build
//---------------------------------------------------------------------------//
/*!
 * Construct for the given global kernel F.
 */
template<class F>
KernelParamCalculator::KernelParamCalculator(F kernel_func, const char* name)
    : KernelParamCalculator(
        kernel_func, name, celeritas::device().default_block_size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct for the given global kernel F.
 */
template<class F>
KernelParamCalculator::KernelParamCalculator(F           kernel_func,
                                             const char* name,
                                             dim_type    block_size)
    : block_size_(block_size)
{
    CELER_EXPECT(block_size % celeritas::device().warp_size() == 0);
    id_ = celeritas::kernel_diagnostics().insert(kernel_func, name, block_size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
