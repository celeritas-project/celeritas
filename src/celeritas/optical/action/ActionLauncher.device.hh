//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ActionLauncher.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Profile and launch optical stepping loop kernels.
 *
 * This is an extension to \c KernelLauncher which uses an action's label and
 * takes the optical state to determine the launch size. The "call thread"
 * operation (thread executor) should contain the params and state.
 *
 * Example:
 * \code
 void FooAction::step(CoreParams const& params,
                      CoreStateDevice& state) const
 {
   auto execute_thread = make_blah_executor(blah);
   static ActionLauncher<decltype(execute_thread)> const launch_kernel(*this);
   launch_kernel(state, execute_thread);
 }
 * \endcode
 */
template<class F>
class ActionLauncher : public KernelLauncher<F>
{
    static_assert(
        (std::is_trivially_copyable_v<F> || CELERITAS_USE_HIP
         || CELER_COMPILER == CELER_COMPILER_CLANG)
            && !std::is_pointer_v<F> && !std::is_reference_v<F>,
        R"(Launched action must be a trivially copyable function object)");
    using StepActionT = OpticalStepActionInterface;

  public:
    // Create a launcher from a string
    using KernelLauncher<F>::KernelLauncher;

    // Create a launcher from an action
    explicit ActionLauncher(StepActionT const& action);

    // Create a launcher with a string extension
    ActionLauncher(StepActionT const& action, std::string_view ext);

    // Launch a kernel for a thread range or number of threads
    using KernelLauncher<F>::operator();

    // Launch a kernel for the wrapped executor
    void operator()(CoreState<MemSpace::device> const& state,
                    F const& call_thread) const;
};

//---------------------------------------------------------------------------//
/*!
 * Create a launcher from an action.
 */
template<class F>
ActionLauncher<F>::ActionLauncher(StepActionT const& action)
    : ActionLauncher{action.label()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Create a launcher with a string extension.
 */
template<class F>
ActionLauncher<F>::ActionLauncher(StepActionT const& action,
                                  std::string_view ext)
    : ActionLauncher{std::string(action.label()) + "-" + std::string(ext)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel for the wrapped executor.
 */
template<class F>
void ActionLauncher<F>::operator()(CoreState<MemSpace::device> const& state,
                                   F const& call_thread) const
{
    return (*this)(
        range(ThreadId{state.size()}), state.stream_id(), call_thread);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
