//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file geocel/rasterize/RaytraceImager.device.t.hh
 * \brief Template definition file for \c RaytraceImager .
 *
 * Include this file in a .cu file and instantiate it explicitly. When
 * instantiating, you must provide access to the GeoTraits specialization as
 * well as the data classes and track view.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "RaytraceImager.hh"

#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/ThreadId.hh"

#include "detail/RaytraceExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the raytrace kernel on device.
 */
template<class G>
void RaytraceImager<G>::launch_raytrace_kernel(
    GeoParamsCRef<MemSpace::device> const& geo_params,
    GeoStateRef<MemSpace::device> const& geo_states,
    ImageParamsCRef<MemSpace::device> const& img_params,
    ImageStateRef<MemSpace::device> const& img_states) const
{
    using CalcId = detail::VolumeIdCalculator;

    detail::RaytraceExecutor<GeoTrackView, CalcId> execute_thread{
        geo_params, geo_states, img_params, img_states, CalcId{}};

    static KernelLauncher<decltype(execute_thread)> const launch_kernel{
        std::string{"raytrace-"} + GeoTraits<G>::name};
    launch_kernel(geo_states.size(), StreamId{}, execute_thread);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
