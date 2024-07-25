//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"

#include "ActionInterface.hh"
#include "CoreParams.hh"
#include "CoreState.hh"
#include "KernelContextException.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper function to run an executor in parallel on CPU.
 *
 * Example:
 * \code
 void FooHelper::execute(CoreParams const& params,
                         CoreStateHost& state) const
 {
    launch_core(params, state, "foo-helper", make_blah_executor(blah));
 }
 * \endcode
 */
template<class F>
void launch_core(std::string_view label,
                 celeritas::CoreParams const& params,
                 celeritas::CoreState<MemSpace::host>& state,
                 F&& execute_thread)
{
    MultiExceptionHandler capture_exception;
#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0, size = state.size(); i != size; ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            execute_thread(ThreadId{i}),
            capture_exception,
            KernelContextException(
                params.ref<MemSpace::host>(), state.ref(), ThreadId{i}, label));
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to run an action in parallel on CPU over all states.
 *
 * Example:
 * \code
 void FooAction::execute(CoreParams const& params,
                         CoreStateHost& state) const
 {
    launch_action(*this, params, state, make_blah_executor(blah));
 }
 * \endcode
 */
template<class F>
void launch_action(ExplicitActionInterface const& action,
                   celeritas::CoreParams const& params,
                   celeritas::CoreState<MemSpace::host>& state,
                   F&& execute_thread)
{
    return launch_core(
        action.label(), params, state, std::forward<F>(execute_thread));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
