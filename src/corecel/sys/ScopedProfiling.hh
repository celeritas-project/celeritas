//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RAII class for scoped profiling.
 *
 * Implementations should support multithreaded context where each thread have
 * one or more alive instance of this class.
 *
 * This is useful for wrapping specific code fragment in a range for profiling,
 * e.g. ignoring of VecGeom instantiation kernels, profiling a specific action
 * or loop on the CPU.
 */
class ScopedProfiling
{
  public:
    // Activate profiling
    explicit ScopedProfiling(std::string const& name);
    // Deactivate profiling
    ~ScopedProfiling();
    // RAII semantics
    void* operator new(std::size_t count) = delete;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline ScopedProfiling::ScopedProfiling(std::string const&) {}
inline ScopedProfiling::~ScopedProfiling() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
