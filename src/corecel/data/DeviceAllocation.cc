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
DeviceAllocation::DeviceAllocation(size_type bytes) : size_(bytes)
{
    CELER_EXPECT(celeritas::device());
    void* ptr = nullptr;
    CELER_DEVICE_CALL_PREFIX(Malloc(&ptr, bytes));
    data_.reset(static_cast<Byte*>(ptr));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
void DeviceAllocation::copy_to_device(SpanConstBytes bytes)
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
void DeviceAllocation::copy_to_host(SpanBytes bytes) const
{
    CELER_EXPECT(bytes.size() == this->size());
    CELER_DEVICE_CALL_PREFIX(Memcpy(bytes.data(),
                                    data_.get(),
                                    this->size(),
                                    CELER_DEVICE_PREFIX(MemcpyDeviceToHost)));
}

//---------------------------------------------------------------------------//
//! Deleter frees data: prevent exceptions
void DeviceAllocation::DeviceFreeDeleter::operator()(
    [[maybe_unused]] Byte* ptr) const
{
    try
    {
        CELER_DEVICE_CALL_PREFIX(Free(ptr));
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
