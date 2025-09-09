//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/KernelLauncherImpl.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"

#include "../KernelParamCalculator.device.hh"
#include "../KernelTraits.hh"
#include "../ThreadId.hh"

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
__global__ void __launch_bounds__(CELERITAS_MAX_BLOCK_SIZE)
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier has no manual launch bounds
template<class F,
         std::enable_if_t<kernel_no_bound<typename F::Applier>, bool> = true>
__global__ void __launch_bounds__(CELERITAS_MAX_BLOCK_SIZE)
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier defines the first launch bounds argument
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks<A_>, bool> = true>
__global__ void __launch_bounds__(A_::max_block_size)
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

// Instantiated if F::Applier defines two arguments for launch bounds
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks_min_warps<A_>, bool> = true>
__global__ void
#if CELERITAS_USE_CUDA
__launch_bounds__(A_::max_block_size,
                  (A_::min_warps_per_eu * 32) / A_::max_block_size)
#elif CELERITAS_USE_HIP
__launch_bounds__(A_::max_block_size, A_::min_warps_per_eu)
#endif
    launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace detail
}  // namespace celeritas
// vim: set ft=cuda :
