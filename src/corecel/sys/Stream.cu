//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.cu
//---------------------------------------------------------------------------//
#include "Stream.hh"

#include <iostream>

#include "corecel/DeviceRuntimeApi.hh"  // IWYU pragma: keep

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"

#include "detail/AsyncMemoryResource.device.hh"

namespace celeritas
{
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
        CELER_DEVICE_CALL_PREFIX(StreamDestroy(impl->stream));
        CELER_LOG_LOCAL(debug)
            << "Destroyed stream " << static_cast<void*>(impl->stream);
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
    CELER_DEVICE_CALL_PREFIX(StreamCreate(&stream));
#if CUDART_VERSION >= 12000
    unsigned long long stream_id = -1;
    CELER_CUDA_CALL(cudaStreamGetId(stream, &stream_id));
    CELER_LOG_LOCAL(debug) << "Created stream ID " << stream_id;
#else
    CELER_LOG_LOCAL(debug) << "Created stream  " << static_cast<void*>(stream);
#endif
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
    CELER_DEVICE_CALL_PREFIX(StreamSynchronize(impl_->stream));
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
