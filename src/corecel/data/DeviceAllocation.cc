//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAllocation.cc
//---------------------------------------------------------------------------//
#include "DeviceAllocation.hh"

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct in unallocated state.
 */
DeviceAllocation::DeviceAllocation(StreamId stream)
    : size_{0}, stream_{stream}, data_{nullptr, {stream}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Allocate a buffer with the given number of bytes.
 */
DeviceAllocation::DeviceAllocation(size_type bytes) : size_{bytes}
{
    CELER_EXPECT(celeritas::device());
    void* ptr = nullptr;
    CELER_DEVICE_API_CALL(Malloc(&ptr, bytes));
    data_.reset(static_cast<std::byte*>(ptr));
}

//---------------------------------------------------------------------------//
/*!
 * Allocate a buffer asynchronously with the given number of bytes.
 */
DeviceAllocation::DeviceAllocation(size_type bytes, StreamId stream)
    : size_{bytes}, stream_{stream}, data_{nullptr, {stream}}
{
    CELER_EXPECT(celeritas::device());
    CELER_EXPECT(stream);
    void* ptr = celeritas::device().stream(stream_).malloc_async(bytes);
    CELER_ASSERT(ptr);
    data_.reset(static_cast<std::byte*>(ptr));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
void DeviceAllocation::copy_to_device(SpanConstBytes bytes)
{
    CELER_EXPECT(bytes.size() <= this->size());
    if (stream_)
    {
        CELER_DEVICE_API_CALL(
            MemcpyAsync(data_.get(),
                        bytes.data(),
                        bytes.size(),
                        CELER_DEVICE_API_SYMBOL(MemcpyHostToDevice),
                        celeritas::device().stream(stream_).get()));
    }
    else
    {
        CELER_DEVICE_API_CALL(
            Memcpy(data_.get(),
                   bytes.data(),
                   bytes.size(),
                   CELER_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
void DeviceAllocation::copy_to_host(SpanBytes bytes) const
{
    CELER_EXPECT(bytes.size() <= this->size());
    if (stream_)
    {
        CELER_DEVICE_API_CALL(
            MemcpyAsync(bytes.data(),
                        data_.get(),
                        this->size(),
                        CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost),
                        celeritas::device().stream(stream_).get()));
    }
    else
    {
        CELER_DEVICE_API_CALL(
            Memcpy(bytes.data(),
                   data_.get(),
                   this->size(),
                   CELER_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
    }
}

//---------------------------------------------------------------------------//
//! Deleter frees data: prevent exceptions
void DeviceAllocation::DeviceFreeDeleter::operator()(
    [[maybe_unused]] std::byte* ptr) const noexcept(CELER_USE_DEVICE)
{
    try
    {
        if (stream_)
        {
            celeritas::device().stream(stream_).free_async(ptr);
        }
        else
        {
            CELER_DEVICE_API_CALL(Free(ptr));
        }
    }
    catch (RuntimeError const& e)
    {
        // The only errors likely from cudaFree is an "unclearable" error
        // message from an earlier kernel failure (assertion or invalid memory
        // access)
        static int warn_count = 0;
        if (warn_count <= 1)
        {
            CELER_LOG(debug) << "While freeing device memory: " << e.what();
        }
        if (warn_count == 1)
        {
            CELER_LOG(debug) << "Suppressing further DeviceFreeDeleter "
                                "warning messages";
        }
        ++warn_count;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
