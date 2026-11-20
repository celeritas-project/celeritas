//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/AsyncMemoryResource.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/mr/memory_resource.h>

#include "AsyncMemoryResource.hh"

namespace celeritas
{
namespace detail
{
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
