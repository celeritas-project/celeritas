//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.cc
//---------------------------------------------------------------------------//
// NOTE: runtime API *must* be included before header
#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep
//------------------------------------//
#include "Stream.hh"
//------------------------------------//
#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "Device.hh"
#include "DeviceEvent.hh"

#include "detail/AsyncMemoryResource.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Safely print a stream's ID (if possible)
struct StreamableStream
{
    Stream::StreamT s;
};

#if !CELER_DEVICE_COMPILE
std::ostream& operator<<(std::ostream& os, StreamableStream const& sds)
{
#    if CUDART_VERSION >= 12000
    unsigned long long stream_id{0};
    CELER_DEVICE_API_CALL(StreamGetId(sds.s, &stream_id));
    os << "id=" << stream_id;
#    else
    os << '@' << static_cast<void*>(sds.s);
#    endif
    return os;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
//! PIMPL class
struct Stream::Impl
{
    StreamT stream{nullptr};
    ResourceT memory_resource;
};

//---------------------------------------------------------------------------//
/*!
 * Destroy the stream.
 */
void Stream::ImplDeleter::operator()(Impl* impl) noexcept
{
    if constexpr (CELER_USE_DEVICE)
    {
        try
        {
            CELER_LOG_LOCAL(debug)
                << "Destroying stream " << StreamableStream{impl->stream};
            CELER_DEVICE_API_CALL(StreamDestroy(impl->stream));
            delete impl;
        }
        catch (RuntimeError const& e)
        {
            std::cerr << "Failed to destroy stream: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Failed to destroy stream" << std::endl;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct by creating a stream with the active device context.
 *
 * \pre A device must be active and configured.
 * \deprecated This constructor is ambiguous: remove it.
 */
Stream::Stream() : Stream(celeritas::device()) {}

//---------------------------------------------------------------------------//
/*!
 * Construct a stream for the given device.
 *
 * This delegates to the default constructor, which creates a stream on the
 * active device. The device reference is used to enforce that a valid device
 * has been activated before stream creation.
 *
 * \pre The device must be valid and active.
 */
Stream::Stream(Device const& device)
{
    CELER_EXPECT(device);
    StreamT stream;
    CELER_DEVICE_API_CALL(StreamCreate(&stream));
    CELER_LOG_LOCAL(debug) << "Created stream " << StreamableStream{stream};
    impl_.reset(new Impl);
    impl_->stream = stream;
    impl_->memory_resource = ResourceT{stream};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a null stream.
 */
Stream::Stream(std::nullptr_t)
{
    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Get the CUDA stream pointer.
 */
Stream::StreamT Stream::get() const
{
    CELER_EXPECT(*this);
    return impl_->stream;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Thrust async allocation resource.
 */
Stream::ResourceT& Stream::memory_resource()
{
    CELER_EXPECT(*this);
    return impl_->memory_resource;
}

//---------------------------------------------------------------------------//
/*!
 * Allocate memory asynchronously on this stream if possible.
 *
 * HIP 5.1 and lower does not support async allocation.
 */
void* Stream::malloc_async(std::size_t bytes)
{
    CELER_EXPECT(*this);
    return detail::malloc_async(bytes, impl_->stream);
}

//---------------------------------------------------------------------------//
/*!
 * Free memory asynchronously on this stream if possible.
 */
void Stream::free_async(void* ptr)
{
    CELER_EXPECT(*this);
    return detail::free_async(ptr, impl_->stream);
}

//---------------------------------------------------------------------------//
/*!
 * Block host execution until stream operations are all complete.
 */
void Stream::sync() const
{
    CELER_EXPECT(*this);
    CELER_DEVICE_API_CALL(StreamSynchronize(impl_->stream));
}

//---------------------------------------------------------------------------//
/*!
 * Block stream execution until the event completes.
 */
void Stream::wait(DeviceEvent const& e)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(e);
    CELER_DEVICE_API_CALL(StreamWaitEvent(this->get(), e.get()));
}

//---------------------------------------------------------------------------//
/*!
 * Enqueue delayed execution of a host function.
 */
void Stream::launch_host_func(HostKernel func, void* data)
{
    CELER_EXPECT(*this);
    CELER_DEVICE_API_CALL(LaunchHostFunc(impl_->stream, func, data));
#if !CELER_USE_DEVICE
    CELER_DISCARD(func);
    CELER_DISCARD(data);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
