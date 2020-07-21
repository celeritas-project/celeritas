//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceAllocation.cuda.cc
//---------------------------------------------------------------------------//
#include "DeviceAllocation.hh"

#include <cuda_runtime_api.h>
#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Allocate a buffer with the given number of bytes.
 */
DeviceAllocation::DeviceAllocation(size_type bytes) : size_(bytes)
{
    REQUIRE(bytes > 0);
    void* ptr = nullptr;
    CELER_CUDA_CALL(cudaMalloc(&ptr, bytes));
    data_.reset(static_cast<byte*>(ptr));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
void DeviceAllocation::copy_to_device(constSpanBytes bytes)
{
    REQUIRE(bytes.size() == this->size());
    CELER_CUDA_CALL(cudaMemcpy(
        data_.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
void DeviceAllocation::copy_to_host(SpanBytes bytes) const
{
    REQUIRE(bytes.size() == this->size());
    CELER_CUDA_CALL(cudaMemcpy(
        bytes.data(), data_.get(), this->size(), cudaMemcpyDeviceToHost));
}

//---------------------------------------------------------------------------//
//! Deleter frees cuda data
void DeviceAllocation::CudaFreeDeleter::operator()(byte* ptr) const
{
    CELER_CUDA_CALL(cudaFree(ptr));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
