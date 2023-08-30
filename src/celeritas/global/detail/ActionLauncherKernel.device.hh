//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionLauncherKernel.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ThreadId.hh"

#include "ApplierTraits.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Celeritas executor kernel implementation.
 */
template<class F>
__device__ CELER_FORCEINLINE void
launch_kernel_impl(Range<ThreadId> const& thread_range, F& execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < thread_range.size()))
        return;
    execute_thread(*(thread_range.cbegin() + tid.get()));
}

//---------------------------------------------------------------------------//
//!@{
//! Launch the given executor using thread ids in the thread_range.

// Instantiated if F doesn't define a member type F::Applier
template<class F, std::enable_if_t<!has_applier_v<F>, bool> = true>
__global__ void
launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier has no launch bounds
template<class F,
         std::enable_if_t<kernel_no_bound<typename F::Applier>, bool> = true>
__global__ void
launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier defines the first launch bounds argument
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks<A_>, bool> = true,
         int T_ = A_::max_block_size>
__global__ void __launch_bounds__(T_)
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier defines two arguments for launch bounds
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks_min_warps<A_>, bool> = true,
         int T_ = A_::max_block_size,
#if CELERITAS_USE_CUDA
         int B_ = (A_::min_warps_per_eu * 32) / T_>
#elif CELERITAS_USE_HIP
         // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#porting-from-cuda-launch-bounds
         int B_ = A_::min_warps_per_eu>
#else
#    error \
        "Compiling device code without setting either CELERITAS_USE_CUDA or CELERITAS_USE_HIP"
#endif
__global__ void __launch_bounds__(T_, B_)
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace detail
}  // namespace celeritas
