//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//! \file corecel/sys/Stream.cu
//---------------------------------------------------------------------------//
#include "Stream.hh"

#include <iostream>

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "Device.hh"

#include "detail/AsyncMemoryResource.device.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Safely print a stream's ID (if possible) or
struct StreamableStream
{
    detail::DeviceStream_t s;
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
// PIMPL class
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

//---------------------------------------------------------------------------//
/*!
 * Construct by creating a stream.
 */
Stream::Stream()
{
    StreamT stream;
    CELER_DEVICE_API_CALL(StreamCreate(&stream));
    CELER_LOG_LOCAL(debug) << "Created stream " << StreamableStream{stream};
    impl_.reset(new Impl);
    impl_->stream = stream;
    impl_->memory_resource = ResourceT{stream};
}

//---------------------------------------------------------------------------//
/*!
 * Get the CUDA stream pointer.
 */
Stream::StreamT Stream::get() const
{
    return impl_->stream;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Thrust async allocation resource.
 */
Stream::ResourceT& Stream::memory_resource()
{
    return impl_->memory_resource;
}

//---------------------------------------------------------------------------//
/*!
 * Synchronize this stream.
 */
void Stream::sync() const
{
    CELER_DEVICE_API_CALL(StreamSynchronize(impl_->stream));
}

//---------------------------------------------------------------------------//
/*!
 * Allocate memory asynchronously on this stream if possible.
 *
 * HIP 5.1 and lower does not support async allocation.
 */
void* Stream::malloc_async(std::size_t bytes) const
{
    return detail::malloc_async(bytes, impl_->stream);
}

//---------------------------------------------------------------------------//
/*!
 * Free memory asynchronously on this stream if possible.
 */
void Stream::free_async(void* ptr) const
{
    return detail::free_async(ptr, impl_->stream);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
