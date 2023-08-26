//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAllocation.cc
//---------------------------------------------------------------------------//
#include "DeviceAllocation.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Allocate a buffer with the given number of bytes.
 */
template<DeviceAllocationPolicy P>
DeviceAllocation<P>::DeviceAllocation(size_type bytes, StreamT stream)
    : size_{bytes}, stream_{stream}, data_{nullptr, DeviceFreeDeleter{stream}}
{
    CELER_EXPECT(celeritas::device());
    void* ptr = nullptr;
    if constexpr (P == DeviceAllocationPolicy::sync)
    {
        CELER_DEVICE_CALL_PREFIX(Malloc(&ptr, bytes));
    }
    else if constexpr (P == DeviceAllocationPolicy::async)
    {
        CELER_DEVICE_CALL_PREFIX(MallocAsync(&ptr, bytes, stream_));
    }
    data_.reset(static_cast<Byte*>(ptr));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<DeviceAllocationPolicy P>
void DeviceAllocation<P>::copy_to_device(SpanConstBytes bytes)
{
    CELER_EXPECT(bytes.size() == this->size());
    CELER_DEVICE_CALL_PREFIX(Memcpy(data_.get(),
                                    bytes.data(),
                                    bytes.size(),
                                    CELER_DEVICE_PREFIX(MemcpyHostToDevice)));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<DeviceAllocationPolicy P>
void DeviceAllocation<P>::copy_to_host(SpanBytes bytes) const
{
    CELER_EXPECT(bytes.size() == this->size());
    CELER_DEVICE_CALL_PREFIX(Memcpy(bytes.data(),
                                    data_.get(),
                                    this->size(),
                                    CELER_DEVICE_PREFIX(MemcpyDeviceToHost)));
}

//---------------------------------------------------------------------------//
//! Deleter frees data: prevent exceptions
template<DeviceAllocationPolicy P>
void DeviceAllocation<P>::DeviceFreeDeleter::operator()(
    [[maybe_unused]] Byte* ptr) const
{
    try
    {
        if constexpr (P == DeviceAllocationPolicy::sync)
        {
            CELER_DEVICE_CALL_PREFIX(Free(ptr));
        }
        else if constexpr (P == DeviceAllocationPolicy::async)
        {
            CELER_DEVICE_CALL_PREFIX(FreeAsync(ptr, stream_));
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

// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class DeviceAllocation<DeviceAllocationPolicy::sync>;
template class DeviceAllocation<DeviceAllocationPolicy::async>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
