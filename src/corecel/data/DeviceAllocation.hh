//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAllocation.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>
#include <memory>
#include <utility>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

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
    using size_type = std::size_t;
    using SpanBytes = Span<Byte>;
    using SpanConstBytes = Span<Byte const>;
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
    struct DeviceFreeDeleter
    {
        void operator()(Byte*) const;
    };
    using DeviceUniquePtr = std::unique_ptr<Byte[], DeviceFreeDeleter>;

    //// DATA ////

    detail::InitializedValue<size_type> size_;
    DeviceUniquePtr data_;
};

// Swap two allocations
inline void swap(DeviceAllocation& a, DeviceAllocation& b) noexcept;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Swap with another allocation.
 */
void DeviceAllocation::swap(DeviceAllocation& other) noexcept
{
    using std::swap;
    swap(this->data_, other.data_);
    swap(this->size_, other.size_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the owned device memory.
 */
auto DeviceAllocation::device_ref() -> SpanBytes
{
    return {data_.get(), size_};
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the owned device memory.
 */
auto DeviceAllocation::device_ref() const -> SpanConstBytes
{
    return {data_.get(), size_};
}

//---------------------------------------------------------------------------//
/*!
 * Swap two allocations.
 */
void swap(DeviceAllocation& a, DeviceAllocation& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
