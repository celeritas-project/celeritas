//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/AsyncMemoryResource.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/mr/memory_resource.h>

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! CUDA/HIP opaque stream handle
using DeviceStream_t = CELER_DEVICE_API_SYMBOL(Stream_t);

//! Allocate memory asynchronously if supported
void* malloc_async(std::size_t bytes, DeviceStream_t s);
//! Free memory asynchronously if supported
void free_async(void* ptr, DeviceStream_t s);

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

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
