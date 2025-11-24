//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionLauncher.device.hh
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
#include "celeritas/track/TrackInitParams.hh"

#include "ActionInterface.hh"
#include "CoreParams.hh"
#include "CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Profile and launch core stepping loop kernels.
 *
 * This is an extension to \c KernelLauncher which uses an action's label and
 * takes core params/state to determine the launch size and/or action range.
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
        Launchable_v<F>,
        R"(Launched action must be a trivially copyable function object)");

    using StepActionT = CoreStepActionInterface;

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
                    F const& execute_thread) const;

    // Launch with reduced grid size for when tracks are sorted
    void operator()(StepActionT const& action,
                    CoreParams const& params,
                    CoreState<MemSpace::device> const& state,
                    F const& execute_thread) const;
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
                                   F const& execute_thread) const
{
    return (*this)(
        range(ThreadId{state.size()}), state.stream_id(), execute_thread);
}

//---------------------------------------------------------------------------//
/*!
 * Launch with reduced grid size for when tracks are sorted.
 *
 * These argument should be consistent with those in \c ActionLauncher.hh .
 */
template<class F>
void ActionLauncher<F>::operator()(StepActionT const& action,
                                   CoreParams const& params,
                                   CoreState<MemSpace::device> const& state,
                                   F const& execute_thread) const
{
    if (state.has_action_range()
        && is_action_sorted(action.order(), params.init()->track_order()))
    {
        // Launch on a subset of threads
        return (*this)(state.get_action_range(action.action_id()),
                       state.stream_id(),
                       execute_thread);
    }
    else
    {
        // Not partitioned by action: launch on all threads
        return (*this)(state, execute_thread);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
