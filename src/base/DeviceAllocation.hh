//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
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
 * Note that \c byte is defined in \c Span.hh as an enum class with type
 * unsigned char.
 */
class DeviceAllocation
{
  public:
    //@{
    //! Type aliases
    using SpanBytes      = span<byte>;
    using constSpanBytes = span<const byte>;
    //@}

  public:
    // Construct in unallocated state
    DeviceAllocation() = default;

    // Construct and allocate a number of bytes
    DeviceAllocation(size_type num_bytes);

    // Swap with another allocation
    inline void swap(DeviceAllocation& other) noexcept;

    // >>> ACCESSORS

    //! Get the number of bytes allocated
    size_type size() const { return size_; }

    //! Whether memory is allocated
    bool empty() const { return size_ == 0; }

    // >>> DEVICE ACCESSORS

    // Get the device pointer
    inline SpanBytes device_pointers();

    // Get the device pointer
    inline constSpanBytes device_pointers() const;

    // Copy data to device
    void copy_to_device(constSpanBytes bytes);

    // Copy data to host
    void copy_to_host(SpanBytes bytes) const;

  private:
    struct CudaFreeDeleter
    {
        void operator()(byte*) const;
    };
    using DeviceUniquePtr = std::unique_ptr<byte[], CudaFreeDeleter>;

    // >>> DATA

    detail::InitializedValue<size_type> size_;
    DeviceUniquePtr data_;
};

// Swap two allocations
inline void swap(DeviceAllocation& a, DeviceAllocation& b) noexcept;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DeviceAllocation.i.hh"

//---------------------------------------------------------------------------//
