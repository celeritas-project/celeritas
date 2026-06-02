//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Openmp.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"

#include "ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

/*!
 * \typedef OpenmpThreadId
 *
 * Type-safe alias based on OpenMP usage:
 * - when using event-level OpenMP parallelism, this is \c StreamId (where each
 *   stream is a separate state vector).
 * - when using track-level OpenMP parallelism, this is \c TrackSlotId .
 *
 * \sa launch_kernel
 */
#if CELERITAS_OPENMP == CELERITAS_OPENMP_EVENT
using OpenmpThreadId = StreamId;
#elif CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
using OpenmpThreadId = TrackSlotId;
#else
using OpenmpThreadId = OpaqueId<OpenmpThread_, unsigned int>;
#endif
using OpenmpSize_t = MakeSize_t<OpenmpThreadId>;

// Get the maximum number of threads that can execute in parallel
OpenmpSize_t openmp_thread_limit();

// Get the maximum number of threads in a new parallel region
OpenmpSize_t openmp_max_threads();

// Get the number of threads in the *current* parallel regino
OpenmpSize_t openmp_num_threads();

// Get a thread ID corresponding to the current OpenMP thread ID
OpenmpThreadId openmp_local_thread();

// Set the maximum number of threads for default future parallel regions
void openmp_num_threads(OpenmpSize_t);

// Get the openmp process bind affinity
char const* openmp_proc_bind();

//---------------------------------------------------------------------------//
}  // namespace celeritas
