//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceAllocation.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "Span.hh"
#include "Types.hh"
#include "detail/InitializedValue.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Allocate raw uninitialized memory.
 *
 * This class is intended to be used by host-compiler `.hh` code as a bridge to
 * device memory. It allows Storage classes to allocate and manage device
 * memory without using `thrust`, which requires NVCC and propagates that
 * requirement into all upstream code.
 */
class DeviceAllocation
{
  public:
    //!@{
    //! Type aliases
    using SpanBytes      = Span<Byte>;
    using SpanConstBytes = Span<const Byte>;
    //!@}

  public:
    // Construct in unallocated state
    DeviceAllocation() = default;

    // Construct and allocate a number of bytes
    DeviceAllocation(size_type num_bytes);

    // Swap with another allocation
    inline void swap(DeviceAllocation& other) noexcept;

    //// ACCESSORS ////

    //! Get the number of bytes allocated
    size_type size() const { return size_; }

    //! Whether memory is allocated
    bool empty() const { return size_ == 0; }

    //// DEVICE ACCESSORS ////

    // Get the device pointer
    inline SpanBytes device_ref();

    // Get the device pointer
    inline SpanConstBytes device_ref() const;

    // Copy data to device
    void copy_to_device(SpanConstBytes bytes);

    // Copy data to host
    void copy_to_host(SpanBytes bytes) const;

  private:
    struct CudaFreeDeleter
    {
        void operator()(Byte*) const;
    };
    using DeviceUniquePtr = std::unique_ptr<Byte[], CudaFreeDeleter>;

    //// DATA ////

    detail::InitializedValue<size_type> size_;
    DeviceUniquePtr                     data_;
};

// Swap two allocations
inline void swap(DeviceAllocation& a, DeviceAllocation& b) noexcept;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DeviceAllocation.i.hh"
