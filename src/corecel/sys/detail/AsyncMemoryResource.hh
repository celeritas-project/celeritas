//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/AsyncMemoryResource.hh
//! \brief Host-compatible declarations for stream-ordered device allocation
//---------------------------------------------------------------------------//
// NOTE: the memory resource class depends on Thrust and is defined in
// AsyncMemoryResource.device.hh .
#pragma once

#include <cstddef>
#include <memory>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! CUDA/HIP opaque stream handle
#if CELER_USE_DEVICE
using DeviceStream_t = CELER_DEVICE_API_SYMBOL(Stream_t);
#else
using DeviceStream_t = std::nullptr_t;
#endif

class AsyncMemoryResource;

//! Delete a memory resource without requiring its complete type
struct AsyncMemoryResourceDeleter
{
    void operator()(AsyncMemoryResource*) const noexcept;
};

//! Owning pointer to a Thrust memory resource
using UPAsyncMemoryResource
    = std::unique_ptr<AsyncMemoryResource, AsyncMemoryResourceDeleter>;

#if CELER_USE_DEVICE
//! Allocate memory asynchronously if supported
void* malloc_async(std::size_t bytes, DeviceStream_t s);
//! Free memory asynchronously if supported
void free_async(void* ptr, DeviceStream_t s);
//! Create a Thrust memory resource associated with a stream
UPAsyncMemoryResource make_async_memory_resource(DeviceStream_t s);
#else
inline void* malloc_async(std::size_t, DeviceStream_t)
{
    CELER_ASSERT_UNREACHABLE();
}
inline void free_async(void*, DeviceStream_t)
{
    CELER_ASSERT_UNREACHABLE();
}
inline UPAsyncMemoryResource make_async_memory_resource(DeviceStream_t)
{
    CELER_ASSERT_UNREACHABLE();
}
inline void AsyncMemoryResourceDeleter::operator()(
    AsyncMemoryResource*) const noexcept
{
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
