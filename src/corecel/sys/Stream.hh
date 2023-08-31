//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.hh
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/mr/memory_resource.h>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
struct MockStream_st;
#endif

//---------------------------------------------------------------------------//
/*!
 * Thrust async memory resource associated with a Stream.
 */
template<class Pointer>
class AsyncMemoryResource final : public thrust::mr::memory_resource<Pointer>
{
  public:
    //!@{
    //! \name Type aliases
    using pointer = Pointer;
    using stream_t = CELER_DEVICE_PREFIX(Stream_t);
    //!@}

    explicit AsyncMemoryResource(stream_t stream) : stream_{stream} {}
    AsyncMemoryResource() : AsyncMemoryResource{nullptr} {}

    pointer do_allocate(std::size_t bytes, std::size_t) override
    {
        void* ret;
        CELER_DEVICE_CALL_PREFIX(MallocAsync(&ret, bytes, stream_));

        return static_cast<pointer>(ret);
    }

    void do_deallocate(pointer p, std::size_t, std::size_t) override
    {
        CELER_DEVICE_CALL_PREFIX(FreeAsync(p, stream_));
    }

  private:
    stream_t stream_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * CUDA or HIP stream.
 *
 * This creates/destroys a stream on construction/destruction.
 */
class Stream
{
  public:
    //!@{
    //! \name Type aliases
#if CELER_USE_DEVICE
    using StreamT = CELER_DEVICE_PREFIX(Stream_t);
#else
    using StreamT = MockStream_st*;
#endif
    using ResourceT = AsyncMemoryResource<void*>;
    //!@}

  public:
    // Construct by creating a stream
    Stream();

    // Construct with the default stream
    Stream(std::nullptr_t) {}

    // Destroy the stream
    ~Stream();

    // Move construct and assign
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;
    void swap(Stream& other) noexcept;

    // Access the stream
    StreamT get() const { return stream_; }

    // Access the thrust resource allocator associated with the stream
    ResourceT& get_memory_resource() const { return memory_resource_; }

  private:
    StreamT stream_{nullptr};
    // Thrust allocator only accept pointers to non-const
    mutable ResourceT memory_resource_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
