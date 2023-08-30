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
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/version.h>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
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
// ENUMERATIONS
//---------------------------------------------------------------------------//
//! Execution semantics for thrust algorithms.
enum class ThrustExecMode
{
    Sync,
    Async,
};

template<class Pointer = void*>
class AsyncMemoryResource final : public thrust::mr::memory_resource<Pointer>
{
  public:
    using pointer = Pointer;
    using StreamT = CELER_DEVICE_PREFIX(Stream_t);
    using BaseT = thrust::mr::memory_resource<Pointer>;

    explicit AsyncMemoryResource(StreamT stream) : stream_{stream} {}

    Pointer do_allocate(std::size_t bytes, std::size_t) override
    {
        void* ret;
        CELER_DEVICE_CALL_PREFIX(MallocAsync(&ret, bytes, stream_));

        return Pointer(ret);
    }

    void do_deallocate(Pointer p, std::size_t, std::size_t) override
    {
        CELER_DEVICE_CALL_PREFIX(FreeAsync(p, stream_));
    }

  private:
    StreamT stream_{nullptr};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Returns an execution policy depending on thrust's version.
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
 * Returns an execution space for the given stream.
 * The template parameter defines whether the algorithm should be executed
 * synchronously or asynchrounously.
 */
template<ThrustExecMode T = ThrustExecMode::Async>
inline auto thrust_execute_on(StreamId stream_id)
{
    if constexpr (T == ThrustExecMode::Sync)
    {
        return thrust_native::par.on(
            celeritas::device().stream(stream_id).get());
    }
    else if constexpr (T == ThrustExecMode::Async)
    {
        using Alloc = thrust::mr::allocator<char, AsyncMemoryResource<void*>>;
        Stream const& stream = celeritas::device().stream(stream_id);
        // FIXME: one memory resource per stream
        static AsyncMemoryResource async_memory_resource{stream.get()};
        return thrust_async_execution_policy()(Alloc(&async_memory_resource))
            .on(stream.get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
