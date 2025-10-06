//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ActionLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Helper function to run an action in parallel on CPU.
 *
 * This allows using a custom number of threads rather than the state size.
 */
template<class F>
void launch_action(size_type num_threads, F&& execute_thread)
{
    MultiExceptionHandler capture_exception;
#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0; i < num_threads; ++i)
    {
        CELER_TRY_HANDLE(execute_thread(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to run an action in parallel on CPU over all states.
 *
 * Example:
 * \code
 void FooAction::step(CoreParams const& params,
                      CoreStateHost& state) const
 {
    launch_action(state, make_blah_executor(params, state, blah));
 }
 * \endcode
 */
template<class F>
void launch_action(CoreState<MemSpace::host>& state, F&& execute_thread)
{
    return launch_action(state.size(), std::forward<F>(execute_thread));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
