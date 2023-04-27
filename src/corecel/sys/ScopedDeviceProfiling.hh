//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedDeviceProfiling.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RAII class for CUDA/HIP profiling flags .
 *
 * This should be instantiated in a single-thread context. If celeritas::device
 * is enabled, it will call \c (cuda|hip)ProfilerStart at construction and
 * \c \\1ProfilerStop at destruction. If device support is disabled, this class
 * is a null-op.
 *
 * This is useful for wrapping the main stepper/transport loop for profiling to
 * allow ignoring of VecGeom instantiation kernels.
 */
class ScopedDeviceProfiling
{
  public:
    // Activate profiling
    ScopedDeviceProfiling();
    // Deactivate profiling
    ~ScopedDeviceProfiling();

  private:
    bool activated_{false};
};

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline ScopedDeviceProfiling::ScopedDeviceProfiling()
{
    (void)sizeof(activated_);
}
inline ScopedDeviceProfiling::~ScopedDeviceProfiling() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
