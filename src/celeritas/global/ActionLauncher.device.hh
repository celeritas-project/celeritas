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
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/KernelLaunchUtils.hh"

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
 * Launch the given executor up to the maximum thread count with a ThreadId
 * offset.
 */
template<class F>
__global__ void launch_action_impl(size_type const num_threads,
                                   ThreadId offset,
                                   F execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < num_threads))
        return;
    execute_thread(tid + offset.get());
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

  public:
    //! Create a launcher from an action
    explicit ActionLauncher(ExplicitActionInterface const& action)
        : calc_launch_params_{action.label(), &launch_action_impl<F>}
        , id_{action.action_id()}
    {
    }

    //! Create a launcher with a string extension
    ActionLauncher(ExplicitActionInterface const& action, std::string_view ext)
        : calc_launch_params_{action.label() + "-" + std::string(ext),
                              &launch_action_impl<F>}
        , id_{action.action_id()}
    {
    }

    //! Launch a kernel for the wrapped executor
    void operator()(CoreState<MemSpace::device> const& state,
                    F const& call_thread) const
    {
        return (*this)(state.size(), state.stream_id(), call_thread);
    }

    //! Launch a Kernel with reduced grid size if tracks are sorted using the
    //! expected track order strategy.
    void operator()(CoreState<MemSpace::device> const& state,
                    CoreParams const& params,
                    TrackOrder expected,
                    F const& call_thread) const
    {
        auto launch_params
            = compute_launch_params(id_, params, state, expected);
        return (*this)(launch_params.num_threads,
                       state.stream_id(),
                       call_thread,
                       launch_params.threads_offset);
    }

    //! Launch a kernel with a custom number of threads
    void operator()(size_type num_threads,
                    StreamId stream_id,
                    F const& call_thread,
                    ThreadId offset = ThreadId{0}) const
    {
        CELER_EXPECT(num_threads > 0);
        CELER_EXPECT(stream_id);
        CELER_DEVICE_PREFIX(Stream_t)
        stream = celeritas::device().stream(stream_id).get();
        auto config = calc_launch_params_(num_threads);
        launch_action_impl<F>
            <<<config.blocks_per_grid, config.threads_per_block, 0, stream>>>(
                num_threads, offset, call_thread);
    }

  private:
    KernelParamCalculator calc_launch_params_;
    ActionId id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
// vim: set ft=cuda
