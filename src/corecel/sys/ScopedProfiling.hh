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
 * Input arguments for the nvtx implementation.
 */
struct ScopedProfilingInput
{
    // Name of the range
    std::string name;
    // ARGB
    uint32_t color{0xFF00FF00};
    // User data
    int32_t payload{0};
    // Category, used to group ranges together
    uint32_t category{0};
};

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
 * TODO: Template ScopedProfiling over profiling backend if we need to add a
 * new one
 */
class ScopedProfiling
{
  public:
    using Input = ScopedProfilingInput;
    // Activate profiling
    explicit ScopedProfiling(Input input);
    // Deactivate profiling
    ~ScopedProfiling();
    // RAII semantics
    void* operator new(std::size_t count) = delete;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline ScopedProfiling::ScopedProfiling(Input) {}
inline ScopedProfiling::~ScopedProfiling() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
