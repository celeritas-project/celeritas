//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file geocel/rasterize/RaytraceImager.cuda.t.hh
 * \brief Template definition file for \c RaytraceImager .
 *
 * Include this file in a .cu file and instantiate it explicitly. When
 * instantiating, you must provide access to the GeoTraits specialization as
 * well as the data classes and track view.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/rasterize/RaytraceImager.t.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class F>
__global__ void __launch_bounds__(CELERITAS_MAX_BLOCK_SIZE)
    raytrace_kernel(ThreadId::size_type num_threads, F execute_thread)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < num_threads))
        return;
    execute_thread(tid);
}

//---------------------------------------------------------------------------//
}  // namespace

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
    using Executor = detail::RaytraceExecutor<GeoTrackView, CalcId>;

    static KernelParamCalculator const calc_launch_params{
        "raytrace", &raytrace_kernel<Executor>};
    auto config = calc_launch_params(geo_states.size());
    raytrace_kernel<Executor>
        <<<config.blocks_per_grid, config.threads_per_block, 0>>>(
            geo_states.size(),
            Executor{geo_params, geo_states, img_params, img_state, CalcId{}});

    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
