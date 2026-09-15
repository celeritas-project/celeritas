//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Openmp.hh
//! \brief Thin wrappers around OpenMP function calls.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Get the maximum number of threads that can execute in parallel
size_type openmp_thread_limit();

// Get the maximum number of threads in a new parallel region
size_type openmp_max_threads();

// Set the maximum number of threads for default future parallel regions
void openmp_num_threads(size_type);

// Get the openmp process bind affinity
char const* openmp_proc_bind();

//---------------------------------------------------------------------------//
}  // namespace celeritas
