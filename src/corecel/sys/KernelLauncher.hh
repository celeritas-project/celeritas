//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 *
 * \par Example
 * \code
 void do_something()
 {
    launch_kernel(num_threads, make_blah_executor(blah));
 }
 * \endcode

 * \todo Rename \c foreach_track_slot and pass \c TrackSlotId ? There's a
 * 1-to-1 correspondence for CUDA (except when track sorting is enabled) but
 * not for CPU threads.
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
