//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.cc
//---------------------------------------------------------------------------//
#include "Stream.hh"

#include <algorithm>
#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"

#if CELERITAS_USE_CUDA
#    define CELER_STREAM_SUPPORTS_ASYNC 1
#elif CELERITAS_USE_HIP       \
    && (HIP_VERSION_MAJOR > 5 \
        || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 2))
#    define CELER_STREAM_SUPPORTS_ASYNC 1
#else
#    define CELER_STREAM_SUPPORTS_ASYNC 0
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Malloc asynchronously for CUDA and newer HIP versions
void* malloc_async_impl(std::size_t bytes, Stream::StreamT s)
{
    void* ptr{};
#if CELER_STREAM_SUPPORTS_ASYNC
    CELER_DEVICE_CALL_PREFIX(MallocAsync(&ptr, bytes, s));
#else
    CELER_DISCARD(bytes);
    CELER_DISCARD(s);
    CELER_DEVICE_CALL_PREFIX(Malloc(&ptr, bytes));
#endif
    return ptr;
}

//---------------------------------------------------------------------------//
//! Free asynchronously for CUDA and newer HIP versions
void free_async_impl(void* ptr, Stream::StreamT s)
{
#if CELER_STREAM_SUPPORTS_ASYNC
    CELER_DEVICE_CALL_PREFIX(FreeAsync(ptr, s));
#else
    CELER_DISCARD(ptr);
    CELER_DISCARD(s);
    CELER_DEVICE_CALL_PREFIX(Free(ptr));
#endif
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Allocate device memory.
 */
template<class Pointer>
auto AsyncMemoryResource<Pointer>::do_allocate(std::size_t bytes, std::size_t)
    -> pointer
{
    return static_cast<pointer>(malloc_async_impl(bytes, stream_));
}

//---------------------------------------------------------------------------//
/*!
 * Deallocate device memory.
 */
template<class Pointer>
void AsyncMemoryResource<Pointer>::do_deallocate(pointer p,
                                                 std::size_t,
                                                 std::size_t)
{
    try
    {
        return free_async_impl(p, stream_);
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
            CELER_LOG(debug) << "Suppressing further AsyncMemoryResource "
                                "warning messages";
        }
        ++warn_count;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct by creating a stream.
 */
Stream::Stream() : memory_resource_(stream_)
{
    CELER_DEVICE_CALL_PREFIX(StreamCreate(&stream_));
#if CUDART_VERSION >= 12000
    unsigned long long stream_id = -1;
    CELER_CUDA_CALL(cudaStreamGetId(stream_, &stream_id));
    CELER_LOG_LOCAL(debug) << "Created stream ID " << stream_id;
#else
    CELER_LOG_LOCAL(debug) << "Created stream  " << static_cast<void*>(stream_);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the stream.
 */
Stream::~Stream()
{
    if (stream_ != nullptr)
    {
        try
        {
            CELER_DEVICE_CALL_PREFIX(StreamDestroy(stream_));
            CELER_LOG_LOCAL(debug)
                << "Destroyed stream " << static_cast<void*>(stream_);
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
 * Move construct.
 */
Stream::Stream(Stream&& other) noexcept
    : memory_resource_{other.memory_resource_}
{
    this->swap(other);
}

//---------------------------------------------------------------------------//
/*!
 * Move assign.
 */
Stream& Stream::operator=(Stream&& other) noexcept
{
    Stream temp(std::move(other));
    this->swap(temp);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Allocate memory asynchronously on this stream if possible.
 *
 * HIP 5.1 and lower does not support async allocation.
 */
void* Stream::malloc_async(std::size_t bytes) const
{
    return malloc_async_impl(bytes, this->get());
}

//---------------------------------------------------------------------------//
/*!
 * Free memory asynchronously on this stream if possible.
 */
void Stream::free_async(void* ptr) const
{
    return free_async_impl(ptr, this->get());
}

//---------------------------------------------------------------------------//
/*!
 * Swap.
 */
void Stream::swap(Stream& other) noexcept
{
    std::swap(stream_, other.stream_);
    std::swap(memory_resource_, other.memory_resource_);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class AsyncMemoryResource<void*>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
