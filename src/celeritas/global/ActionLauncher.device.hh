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
template<class F, int T, int B = 1, int B_FINAL = B>
#elif CELERITAS_USE_HIP
// see
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#porting-from-cuda-launch-bounds
template<class F, int T, int B = 1, int B_FINAL = (B * T) / 32>
#else
static_assert(false,
              "Compiling device code without setting either "
              "CELERITAS_USE_CUDA or CELERITAS_USE_HIP");
#endif
__global__ void __launch_bounds__(T, B_FINAL)
    launch_bounded_action_impl(Range<ThreadId> const thread_range,
                               F execute_thread)
{
    launch_kernel_impl(thread_range, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Profile and launch Celeritas kernels from inside an action.
 * Additionals - up to two - int template arguments can be specified that will
 * be forwarded to \c __launch_bounds__ to constraint kernel registers usages.
 *
 * Semantics of \c __launch_bounds__ 2nd argument differs between CUDA and HIP.
 * \c ActionLauncher expects Cuda semantics. If Celeritas is built targeting
 * HIP, it will automatically convert that argument to match HIP semantics.
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
template<class F, int... bounds>
class ActionLauncher
{
    static_assert((std::is_trivially_copyable_v<F> || CELERITAS_USE_HIP)
                      && !std::is_pointer_v<F> && !std::is_reference_v<F>,
                  "Launched action must be a trivially copyable function "
                  "object");
    static_assert(sizeof...(bounds) <= 2,
                  "ActionLauncher expects two or less variadic template "
                  "arguments");

  private:
    using kernel_func_ptr_t = void (*)(Range<ThreadId> const, F);

    // TODO: better way to conditionnaly init
    static constexpr kernel_func_ptr_t kernel_func_ptr = [] {
        if constexpr (sizeof...(bounds) == 0)
            return &launch_action_impl<F>;
        else
            return &launch_bounded_action_impl<F, bounds...>;
    }();

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
