//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.cc
//---------------------------------------------------------------------------//
#include "Stream.hh"

#include <algorithm>
#include <iostream>

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"

namespace celeritas
{

template<class Pointer>
auto AsyncMemoryResource<Pointer>::do_allocate(std::size_t bytes, std::size_t)
    -> pointer
{
    void* ret;
    CELER_DEVICE_CALL_PREFIX(MallocAsync(&ret, bytes, stream_));

    return static_cast<pointer>(ret);
}

template<class Pointer>
void AsyncMemoryResource<Pointer>::do_deallocate(pointer p,
                                                 std::size_t,
                                                 std::size_t)
{
    CELER_DEVICE_CALL_PREFIX(FreeAsync(p, stream_));
}

//---------------------------------------------------------------------------//
/*!
 * Construct by creating a stream.
 */
Stream::Stream() : memory_resource_(stream_)
{
    CELER_DEVICE_CALL_PREFIX(StreamCreate(&stream_));
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
