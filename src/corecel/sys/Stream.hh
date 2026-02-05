//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <memory>

#include "corecel/Assert.hh"  // IWYU pragma: keep
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
class Device;
class DeviceEvent;

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
 * \par States
 * - \b Constructed: A valid stream created with an active device. The stream
 *   can be used for device operations.
 * - \b Null: Explicitly constructed with \c nullptr. No stream is created,
 *   but the object is in a valid null state. Operations are no-ops.
 *
 * \internal
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
#if !CELER_USE_DEVICE
    //! Stream implementation is unavailable
    using StreamT = std::nullptr_t;
    using ResourceT = std::nullptr_t;
#elif !defined(CELER_DEVICE_RUNTIME_INCLUDED)
    //! Sentinel type to indicate compilation error
    using MissingDeviceRuntime = void;
    using ResourceT = detail::AsyncMemoryResource;
#else
    //! Actual CUDA/HIP stream opaque pointer
    using StreamT = CELER_DEVICE_API_SYMBOL(Stream_t);
    using ResourceT = detail::AsyncMemoryResource;
#endif
    using HostKernel = void (*)(void*);
    //!@}

  public:
    // Construct by creating a stream
    Stream();
    // Construct a null stream
    Stream(std::nullptr_t);
    // Construct a stream for the given device
    explicit Stream(Device const& device);
    CELER_DEFAULT_MOVE_DELETE_COPY(Stream);
    ~Stream() = default;

    // Whether the stream is valid (not null or moved-from)
    inline explicit operator bool() const;

#if defined(CELER_DEVICE_RUNTIME_INCLUDED) || !CELER_USE_DEVICE
    // Access the stream
    StreamT get() const;
#else
    // Since CUDA/HIP declarations are unavailable, this function cannot be
    // used. Include "corecel/DeviceRuntimeApi.hh" to fix.
    MissingDeviceRuntime get() const {}
#endif

    // Access the thrust resource allocator associated with the stream
    ResourceT& memory_resource();

    // Allocate memory asynchronously on this stream if possible
    void* malloc_async(std::size_t bytes);

    // Free memory asynchronously on this stream if possible
    void free_async(void* ptr);

    // Block host execution until stream operations are all done
    void sync() const;

    // Block stream execution until the event completes
    void wait(DeviceEvent const& d);

    // Delayed execution of a host function
    void launch_host_func(HostKernel func, void* data);

  private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

//---------------------------------------------------------------------------//
/*!
 * Whether the stream is valid (not null or moved-from).
 */
Stream::operator bool() const
{
    return static_cast<bool>(impl_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
