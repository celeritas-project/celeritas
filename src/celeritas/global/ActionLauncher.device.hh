//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "ActionInterface.hh"
#include "ApplierInterface.hh"
#include "CoreParams.hh"
#include "CoreState.hh"
#include "KernelContextException.hh"

namespace celeritas
{
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
template<class F, int T, int B = 1, int __B = (B * 32) / T>
#elif CELERITAS_USE_HIP
// see
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#porting-from-cuda-launch-bounds
template<class F, int T, int B = 1, int __B = B>
#else
#    error \
        "Compiling device code without setting either CELERITAS_USE_CUDA or CELERITAS_USE_HIP"
#endif
__global__ void __launch_bounds__(T, __B)
    launch_bounded_action_impl(Range<ThreadId> const thread_range,
                               F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

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
         class __A = typename F::Applier,
         std::enable_if_t<kernel_no_bound<__A>, bool> = true>
constexpr auto select_kernel() -> decltype(&launch_action_impl<F>)
{
    constexpr auto ptr = &launch_action_impl<F>;
    return ptr;
}

// instantiated if F::Applier defines one argument for launch bound
template<class F,
         class __A = typename F::Applier,
         std::enable_if_t<kernel_max_blocks<__A>, bool> = true>
constexpr auto select_kernel()
    -> decltype(&launch_bounded_action_impl<F, __A::max_block_size>)
{
    constexpr auto ptr = &launch_bounded_action_impl<F, __A::max_block_size>;
    return ptr;
}

// instantiated if F::Applier defines two arguments for launch bounds
template<class F,
         class __A = typename F::Applier,
         std::enable_if_t<kernel_max_blocks_min_warps<__A>, bool> = true>
constexpr auto select_kernel()
    -> decltype(&launch_bounded_action_impl<F,
                                            __A::max_block_size,
                                            __A::min_warps_per_eu>)
{
    constexpr auto ptr
        = &launch_bounded_action_impl<F, __A::max_block_size, __A::min_warps_per_eu>;
    return ptr;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Profile and launch Celeritas kernels from inside an action.
 * The template argument F may define a member type named \c Applier.
 *
 * \c F::Applier should have up to two static constexpr int variables named
 * max_block_size and/or min_warps_per_eu.
 * If present, the kernel will use appropriate \c __launch_bounds__.
 * If \c F::Applier::min_warps_per_eu exists then \c F::Applier::max_block_size
 * must also be present or we get a compile error.
 *
 * Semantics of \c __launch_bounds__ 2nd argument differs between CUDA and HIP.
 * \c ActionLauncher expects HIP semantics. If Celeritas is built targeting
 * CUDA, it will automatically convert that argument to match CUDA semantics.
 *
 * The CUDA-specific 3rd argument \c maxBlocksPerCluster is not supported.
 *
 * Example:
 * \code
 void FooAction::execute(CoreParams const& params,
                         CoreStateDevice& state) const
 {
    auto execute_thread = make_blah_executor(blah);
    static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
    launch_kernel(state, execute_thread);
 }
 * \endcode
 */
template<class F>
class ActionLauncher
{
    static_assert((std::is_trivially_copyable_v<F> || CELERITAS_USE_HIP)
                      && !std::is_pointer_v<F> && !std::is_reference_v<F>,
                  "Launched action must be a trivially copyable function "
                  "object");

  private:
    using kernel_func_ptr_t = void (*)(Range<ThreadId> const, F);

    // TODO: better way to conditionally constexpr init?
    static constexpr kernel_func_ptr_t kernel_func_ptr = select_kernel<F>();

  public:
    //! Create a launcher from an action
    explicit ActionLauncher(ExplicitActionInterface const& action)
        : calc_launch_params_{action.label(), kernel_func_ptr}
    {
    }

    //! Create a launcher with a string extension
    ActionLauncher(ExplicitActionInterface const& action, std::string_view ext)
        : calc_launch_params_{action.label() + "-" + std::string(ext),
                              kernel_func_ptr}
    {
    }

    //! Launch a kernel for a thread range
    void operator()(Range<ThreadId> threads,
                    StreamId stream_id,
                    F const& call_thread) const
    {
        if (!threads.empty())
        {
            CELER_DEVICE_PREFIX(Stream_t)
            stream = celeritas::device().stream(stream_id).get();
            auto config = calc_launch_params_(threads.size());
            (*kernel_func_ptr)<<<config.blocks_per_grid,
                                 config.threads_per_block,
                                 0,
                                 stream>>>(threads, call_thread);
        }
    }

    //! Launch a kernel for the wrapped executor
    void operator()(CoreState<MemSpace::device> const& state,
                    F const& call_thread) const
    {
        return (*this)(
            range(ThreadId{state.size()}), state.stream_id(), call_thread);
    }

    //! Launch a kernel with a custom number of threads
    void operator()(size_type num_threads,
                    StreamId stream_id,
                    F const& call_thread) const
    {
        CELER_EXPECT(num_threads > 0);
        CELER_EXPECT(stream_id);
        (*this)(range(ThreadId{num_threads}), stream_id, call_thread);
    }

    //! Launch a Kernel with reduced grid size if tracks are sorted using the
    //! expected track order strategy.
    //! TODO: Always use an ActionLauncher instance with the action passed as
    //! constructor argument
    void operator()(CoreParams const& params,
                    CoreState<MemSpace::device> const& state,
                    ExplicitActionInterface const& action,
                    F const& call_thread) const
    {
        CELER_EXPECT(state.stream_id());
        if (is_action_sorted(action.order(),
                             params.init()->host_ref().track_order))
        {
            return (*this)(state.get_action_range(action.action_id()),
                           state.stream_id(),
                           call_thread);
        }
        else
        {
            return (*this)(
                range(ThreadId{state.size()}), state.stream_id(), call_thread);
        }
    }

  private:
    KernelParamCalculator calc_launch_params_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
// vim: set ft=cuda
