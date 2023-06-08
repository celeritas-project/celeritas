//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <type_traits>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/detail/ActionStateLauncherImpl.hh"
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
 * Launch the given executor up to the maximum thread count.
 */
template<class F>
__global__ void
launch_action_impl(size_type const num_threads, F execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < num_threads))
        return;
    execute_thread(tid);
}

/*!
 * Launch the given executor using thread ids in the thread_range
 */
template<class F>
__global__ void
launch_action_impl(Range<ThreadId> const thread_range, F execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < thread_range.size()))
        return;
    execute_thread(*(thread_range.cbegin() + tid.get()));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Profile and launch Celeritas kernels from inside an action.
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

    // Function pointer types to specify the correct overloaded kernel
    using KernelNumThreads = void (*)(size_type const, F);

    using KernelThreadRange = void (*)(Range<ThreadId> const, F);

    using KernelFunction
        = std::function<void(CoreState<MemSpace::device> const&, F const&)>;

  public:
    //! Create a launcher from an action
    explicit ActionLauncher(ExplicitActionInterface const& action)
        : calc_launch_params_{init_kernel_param_calculator(action.label(),
                                                           false)}
        , action_id_{action.action_id()}
        , invoke_kernel_{bind_invoke_kernel(false)}
    {
    }

    //! Create a launcher with a string extension
    ActionLauncher(ExplicitActionInterface const& action, std::string_view ext)
        : calc_launch_params_{init_kernel_param_calculator(
            action.label() + "-" + std::string(ext), false)}
        , action_id_{action.action_id()}
        , invoke_kernel_{bind_invoke_kernel(false)}
    {
    }

    //! Create a launcher from an action, minimizing kernel grid size if
    //! sorting is done for this action
    explicit ActionLauncher(CoreParams const& params,
                            ExplicitActionInterface const& action)
        : ActionLauncher(
            params,
            action,
            action.label(),
            action.order() == ActionOrder::post
                    && params.init()->host_ref().track_order
                           == TrackOrder::sort_step_limit_action
                || action.order() == ActionOrder::along
                       && params.init()->host_ref().track_order
                              == TrackOrder::sort_along_step_action)
    {
    }

    //! Create a launcher with a string extension, minimizing kernel grid size
    //! if sorting is done for this action
    ActionLauncher(CoreParams const& params,
                   ExplicitActionInterface const& action,
                   std::string_view ext)
        : ActionLauncher(
            params,
            action,
            action.label() + "-" + std::string(ext),
            action.order() == ActionOrder::post
                    && params.init()->host_ref().track_order
                           == TrackOrder::sort_step_limit_action
                || action.order() == ActionOrder::along
                       && params.init()->host_ref().track_order
                              == TrackOrder::sort_along_step_action)
    {
    }

  private:
    ActionLauncher(CoreParams const& params,
                   ExplicitActionInterface const& action,
                   std::string_view name,
                   bool is_action_sorted)
        : calc_launch_params_{init_kernel_param_calculator(name,
                                                           is_action_sorted)}
        , action_id_{action.action_id()}
        , invoke_kernel_{bind_invoke_kernel(is_action_sorted)}
    {
    }

  public:
    KernelParamCalculator
    init_kernel_param_calculator(std::string_view name,
                                 bool const is_action_sorted)
    {
        if (!is_action_sorted)
        {
            return {name, KernelNumThreads{&launch_action_impl<F>}};
        }
        else
        {
            return {name, KernelThreadRange{&launch_action_impl<F>}};
        }
    }

    //! Select the correct kernel call if sorting is done for this action
    KernelFunction bind_invoke_kernel(bool const is_action_sorted)
    {
        if (!is_action_sorted)
        {
            return [this](CoreState<MemSpace::device> const& state,
                          F const& call_thread) {
                return (*this)(state.size(), state.stream_id(), call_thread);
            };
        }
        else
        {
            return [this](CoreState<MemSpace::device> const& state,
                          F const& call_thread) {
                if (Range<ThreadId> threads
                    = detail::compute_launch_params(action_id_, state);
                    threads.size())
                {
                    CELER_DEVICE_PREFIX(Stream_t)
                    stream
                        = celeritas::device().stream(state.stream_id()).get();
                    auto config = calc_launch_params_(threads.size());
                    launch_action_impl<F>
                        <<<config.blocks_per_grid, config.threads_per_block, 0, stream>>>(
                            threads, call_thread);
                }
            };
        }
    }

    //! Launch a Kernel with reduced grid size if tracks are sorted using the
    //! expected track order strategy.
    void operator()(CoreState<MemSpace::device> const& state,
                    F const& call_thread) const
    {
        CELER_EXPECT(state.stream_id());
        invoke_kernel_(state, call_thread);
    }

    //! Launch a kernel with a custom number of threads
    void operator()(size_type num_threads,
                    StreamId stream_id,
                    F const& call_thread) const
    {
        CELER_EXPECT(num_threads > 0);
        CELER_EXPECT(stream_id);
        CELER_DEVICE_PREFIX(Stream_t)
        stream = celeritas::device().stream(stream_id).get();
        auto config = calc_launch_params_(num_threads);
        launch_action_impl<F>
            <<<config.blocks_per_grid, config.threads_per_block, 0, stream>>>(
                num_threads, call_thread);
    }

  private:
    KernelParamCalculator calc_launch_params_;
    ActionId action_id_;
    KernelFunction invoke_kernel_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
// vim: set ft=cuda
