//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedDeviceProfiling.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "celeritas_config.h"

#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RAII class for CUDA/HIP profiling flags .
 *
 * This should be instantiated in a single-thread context. If celeritas::device
 * is enabled, it will call \c cudaProfilerStart at construction and
 * \c cudaProfilerStop at destruction. If device support is disabled, this
 * class is a null-op. HIP support for the profiler start/stop macros is
 * deprecated so we do not support it at present.
 *
 * This is useful for wrapping the main stepper/transport loop for profiling to
 * allow ignoring of VecGeom instantiation kernels.
 */
class ScopedDeviceProfiling
{
  public:
    // Activate profiling
    ScopedDeviceProfiling(std::string_view name);
    // Deactivate profiling
    ~ScopedDeviceProfiling();
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
inline ScopedDeviceProfiling::ScopedDeviceProfiling(std::string_view) {}
inline ScopedDeviceProfiling::~ScopedDeviceProfiling() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
