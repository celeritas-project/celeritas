//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/Thrust.device.hh
//! \brief Platform and version-specific thrust setup
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/version.h>

#include "celeritas_config.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
#if CELERITAS_USE_CUDA
namespace thrust_native = thrust::cuda;
#elif CELERITAS_USE_HIP
namespace thrust_native = thrust::hip;
#endif

//---------------------------------------------------------------------------//
/*!
 * Returns an execution policy depending on thrust's version
 */
inline auto& thrust_async_execution_policy()
{
#if THRUST_MAJOR_VERSION == 1 && THRUST_MINOR_VERSION < 16
    return thrust_native::par;
#else
    return thrust_native::par_nosync;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Returns an execution space for the given stream
 */
inline auto thrust_execute_on(StreamId stream_id)
{
    return thrust_native::par.on(celeritas::device().stream(stream_id).get());
}

//---------------------------------------------------------------------------//
/*!
 * Returns an execution space for the given stream, executing asynchronously if
 * possible
 */
inline auto thrust_async_execute_on(StreamId stream_id)
{
    return thrust_async_execution_policy().on(
        celeritas::device().stream(stream_id).get());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
