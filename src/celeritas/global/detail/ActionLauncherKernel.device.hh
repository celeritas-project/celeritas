//----------------------------------*-C++-*----------------------------------//
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
//---------------------------------------------------------------------------//
namespace detail
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Celeritas executor kernel implementation.
 */

template<class F>
__device__ CELER_FORCEINLINE void
launch_kernel_impl(Range<ThreadId> const thread_range, F& execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < thread_range.size()))
        return;
    execute_thread(*(thread_range.cbegin() + tid.get()));
}

//---------------------------------------------------------------------------//
/*!
 * Launch the given executor using thread ids in the thread_range
 */

template<class F>
__global__ void
launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the given executor using thread ids in the thread_range with
 \c __launch_bounds__
 */

#if CELERITAS_USE_CUDA
template<class F, int T, int B = 1, int B_ = (B * 32) / T>
#elif CELERITAS_USE_HIP
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#porting-from-cuda-launch-bounds
template<class F, int T, int B = 1, int B_ = B>
#else
#    error \
        "Compiling device code without setting either CELERITAS_USE_CUDA or CELERITAS_USE_HIP"
#endif
__global__ void __launch_bounds__(T, B_)
    launch_bounded_action_impl(Range<ThreadId> const thread_range,
                               F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}
//---------------------------------------------------------------------------//
}  // namespace

// Select the correct kernel at compile time depending on the executor's launch
// bounds.

// instantiated if F doesn't define a member type F::Applier
template<class F, std::enable_if_t<!has_applier_v<F>, bool> = true>
constexpr auto select_kernel() -> decltype(&launch_action_impl<F>)
{
    constexpr auto ptr = &launch_action_impl<F>;
    return ptr;
}

// instantiated if F::Applier has no launch bounds
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_no_bound<A_>, bool> = true>
constexpr auto select_kernel() -> decltype(&launch_action_impl<F>)
{
    constexpr auto ptr = &launch_action_impl<F>;
    return ptr;
}

// instantiated if F::Applier defines one argument for launch bounds
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks<A_>, bool> = true>
constexpr auto select_kernel()
    -> decltype(&launch_bounded_action_impl<F, A_::max_block_size>)
{
    constexpr auto ptr = &launch_bounded_action_impl<F, A_::max_block_size>;
    return ptr;
}

// instantiated if F::Applier defines two arguments for launch bounds
template<class F,
         class A_ = typename F::Applier,
         std::enable_if_t<kernel_max_blocks_min_warps<A_>, bool> = true>
constexpr auto select_kernel()
    -> decltype(&launch_bounded_action_impl<F,
                                            A_::max_block_size,
                                            A_::min_warps_per_eu>)
{
    constexpr auto ptr
        = &launch_bounded_action_impl<F, A_::max_block_size, A_::min_warps_per_eu>;
    return ptr;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas