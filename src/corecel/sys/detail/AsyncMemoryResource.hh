//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/AsyncMemoryResource.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

#if CELER_USE_DEVICE
#    include <thrust/mr/memory_resource.h>
#endif
#include <cstdlib>

#include "corecel/DeviceRuntimeApi.hh"

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

//! Allocate memory asynchronously if supported
void* malloc_async(std::size_t bytes, DeviceStream_t s);
//! Free memory asynchronously if supported
void free_async(void* ptr, DeviceStream_t s);

#if CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Thrust async memory resource associated with a CUDA/HIP stream.
 */
class AsyncMemoryResource final : public thrust::mr::memory_resource<void*>
{
  public:
    //!@{
    //! \name Type aliases
    using pointer = void*;
    using StreamT = DeviceStream_t;
    //!@}

    // Construct memory resource for the stream
    explicit AsyncMemoryResource(StreamT stream) : stream_{stream} {}

    // Construct with default Stream
    AsyncMemoryResource() = default;

    // Allocate device memory
    pointer do_allocate(std::size_t bytes, std::size_t) final;

    // Deallocate device memory
    void do_deallocate(pointer p, std::size_t, std::size_t) final;

  private:
    StreamT stream_{nullptr};
};
#else
class AsyncMemoryResource;
inline void* malloc_async(std::size_t, DeviceStream_t)
{
    CELER_ASSERT_UNREACHABLE();
}
inline void free_async(void*, DeviceStream_t)
{
    CELER_ASSERT_UNREACHABLE();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
