//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//! \file corecel/sys/detail/AsyncMemoryResource.cu
//---------------------------------------------------------------------------//
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

#include "AsyncMemoryResource.device.hh"
#include "../Device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Malloc asynchronously for CUDA and newer HIP versions
void* malloc_async(std::size_t bytes, DeviceStream_t s)
{
    void* ptr{};
    if (Device::async())
    {
#if CELER_STREAM_SUPPORTS_ASYNC
        CELER_DEVICE_API_CALL(MallocAsync(&ptr, bytes, s));
#else
        CELER_DISCARD(ptr);
        CELER_DISCARD(bytes);
        CELER_DISCARD(s);
        CELER_ASSERT_UNREACHABLE();
#endif
    }
    else
    {
        CELER_DEVICE_API_CALL(Malloc(&ptr, bytes));
    }
    return ptr;
}

//---------------------------------------------------------------------------//
//! Free asynchronously for CUDA and newer HIP versions
void free_async(void* ptr, DeviceStream_t s)
{
    if (Device::async())
    {
#if CELER_STREAM_SUPPORTS_ASYNC
        CELER_DEVICE_API_CALL(FreeAsync(ptr, s));
#else
        CELER_DISCARD(ptr);
        CELER_DISCARD(s);
        CELER_ASSERT_UNREACHABLE();
#endif
    }
    else
    {
        CELER_DEVICE_API_CALL(Free(ptr));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Allocate device memory.
 */
auto AsyncMemoryResource::do_allocate(std::size_t bytes, std::size_t) -> pointer
{
    return static_cast<pointer>(malloc_async(bytes, stream_));
}

//---------------------------------------------------------------------------//
/*!
 * Deallocate device memory.
 */
void AsyncMemoryResource::do_deallocate(pointer p, std::size_t, std::size_t)
{
    try
    {
        return free_async(p, stream_);
    }
    catch (RuntimeError const& e)
    {
        static int warn_count = 0;
        if (warn_count <= 1)
        {
            CELER_LOG(debug) << "While freeing device memory: " << e.what();
        }
        if (warn_count == 1)
        {
            CELER_LOG(debug)
                << R"(Suppressing further AsyncMemoryResource warning messages)";
        }
        ++warn_count;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
