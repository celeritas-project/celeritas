//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Thrust.device.hh
//! \brief Platform and version-specific thrust setup
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/mr/allocator.h>
#include <thrust/version.h>

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "Device.hh"
#include "Stream.hh"
#include "ThreadId.hh"

#include "detail/AsyncMemoryResource.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the Thrust synchronous parallel policy.
 */
inline auto& thrust_execute()
{
    return thrust::CELER_DEVICE_PLATFORM::par;
}

//---------------------------------------------------------------------------//
/*!
 * Get a Thrust asynchronous parallel policy for the given stream.
 *
 * For older versions of thrust, this executes synchronously on the stream.
 */
inline auto thrust_execute_on(StreamId stream_id)
{
    using Alloc = thrust::mr::allocator<char, Stream::ResourceT>;
    Stream& stream = celeritas::device().stream(stream_id);
#if THRUST_VERSION >= 101600
    // Newer thrust supports asynchronous par
    auto& par_nosync = thrust::CELER_DEVICE_PLATFORM::par_nosync;
#else
    // Fall back to synchronous execution
    auto& par_nosync = thrust::CELER_DEVICE_PLATFORM::par;
#endif
    return par_nosync(Alloc(&stream.memory_resource())).on(stream.get());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
