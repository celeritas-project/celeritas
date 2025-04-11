//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file geocel/rasterize/RaytraceImager.nocuda.t.hh
 * \brief Template definitions for \c RaytraceImager when CUDA is unsupported
 *
 * If a particular geometry does not support device raytracing, include this
 * file alongside \c RaytraceImager.t.hh before instantiating \c
 * RaytraceImager.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "RaytraceImager.hh"

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace celeritas
{
#if CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Launch the raytrace kernel on device.
 */
template<class G>
void RaytraceImager<G>::launch_raytrace_kernel(
    GeoParamsCRef<MemSpace::device> const&,
    GeoStateRef<MemSpace::device> const&,
    ImageParamsCRef<MemSpace::device> const&,
    ImageStateRef<MemSpace::device> const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
