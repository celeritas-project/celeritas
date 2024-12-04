//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Config.hh"

#include "MultiExceptionHandler.hh"
#include "ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper function to run an executor in parallel on CPU.
 *
 * The function should be an executor with the signature
 * \code void(*)(ThreadId) \endcode .
 *
 * Example:
 * \code
 void do_something()
 {
    launch_kernel(num_threads, make_blah_executor(blah));
 }
 * \endcode
 */
template<class F>
void launch_kernel(size_type num_threads, F&& execute_thread)
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
}  // namespace celeritas
