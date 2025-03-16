//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <memory>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"  // IWYU pragma: keep

#if CELER_DEVICE_SOURCE
#    include "corecel/DeviceRuntimeApi.hh"
#endif

#if CELERITAS_USE_CUDA
#    define CELER_STREAM_SUPPORTS_ASYNC 1
#elif CELERITAS_USE_HIP       \
    && (HIP_VERSION_MAJOR > 5 \
        || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2))
#    define CELER_STREAM_SUPPORTS_ASYNC 1
#else
//! Whether CUDA/HIP is enabled and new enough to support async operations
#    define CELER_STREAM_SUPPORTS_ASYNC 0
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace detail
{
class AsyncMemoryResource;
}

//---------------------------------------------------------------------------//
/*!
 * PIMPL class for CUDA or HIP stream.
 *
 * This creates/destroys a stream on construction/destruction and provides
 * accessors to low-level stream-related functionality. This class will
 * typically be accessed only by low-level device implementations or advanced
 * kernels that need to interact with the device stream.
 *
 * \warning This class interface changes based on available headers.
 * Because the CUDA/HIP stream type are only defined when those paths are
 * included and available (which isn't true for all Celeritas code) we hide the
 * stream unless DeviceRuntimeApi has been included or the file is being
 * compiled by a CUDA/HIP compiler. Worse, our stream class has to manage a
 * Thrust memory resource, and in newer versions of rocthrust, its headers
 * cannot be included at all by a non-HIP compiler. Because this class forward
 * declares the memory resource, downstream uses must include \c
 * corecel/sys/detail/AsyncMemoryResource.device.hh .
 */
class Stream
{
  public:
    //!@{
    //! \name Type aliases
#ifdef CELER_DEVICE_RUNTIME_INCLUDED
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
#else
    using MissingDeviceRuntime = void;
#endif
    using ResourceT = detail::AsyncMemoryResource;
    //!@}

  public:
    // Construct by creating a stream
    Stream();
    CELER_DEFAULT_MOVE_DELETE_COPY(Stream);
    ~Stream() = default;

#ifdef CELER_DEVICE_RUNTIME_INCLUDED
    // Access the stream
    StreamT get() const;
#else
    // Since CUDA/HIP declarations are unavailable, this function cannot be
    // used. Include "corecel/DeviceRuntimeApi.hh" to fix.
    MissingDeviceRuntime get() const {}
#endif

    // Access the thrust resource allocator associated with the stream
    ResourceT& memory_resource();

    // Synchronize this stream
    void sync() const;

    // Allocate memory asynchronously on this stream if possible
    void* malloc_async(std::size_t bytes) const;

    // Free memory asynchronously on this stream if possible
    void free_async(void* ptr) const;

  private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline Stream::Stream()
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

#    ifdef CELER_DEVICE_RUNTIME_INCLUDED
inline Stream::StreamT Stream::get() const
{
    CELER_ASSERT_UNREACHABLE();
}
#    endif

inline Stream::ResourceT& Stream::memory_resource()
{
    CELER_ASSERT_UNREACHABLE();
}

inline void Stream::sync() const
{
    CELER_ASSERT_UNREACHABLE();
}
inline void* Stream::malloc_async(std::size_t) const
{
    CELER_ASSERT_UNREACHABLE();
}

inline void Stream::free_async(void*) const
{
    CELER_ASSERT_UNREACHABLE();
}

inline void Stream::ImplDeleter::operator()(Impl*) noexcept
{
    CELER_UNREACHABLE;
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
