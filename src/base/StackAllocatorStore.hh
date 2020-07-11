//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorStore.hh
//---------------------------------------------------------------------------//
#ifndef base_StackAllocatorStore_hh
#define base_StackAllocatorStore_hh

#include <memory>
#include "DeviceAllocation.hh"
#include "StackAllocatorView.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage a chunk of device memory as an in-kernel stack.
 *
 * This low-level class should be used by other containers to ensure alignment
 * requirements are met.
 */
class StackAllocatorStore
{
  public:
    //@{
    //! Public types
    using size_type = StackAllocatorView::size_type;
    //@}
  public:
    // Construct without data assignment
    StackAllocatorStore() = default;

    // Construct with the number of bytes to allocate on device
    explicit StackAllocatorStore(size_type capacity);

    // >>> HOST ACCESSORS

    //! Size (in bytes) of the allocation
    size_type capacity() const { return capacity_; }

    // Clear allocated data
    void clear();

    // Swap with another stack allocator
    void swap(StackAllocatorStore& other);

    // >>> DEVICE ACCESSORS

    // Get a view to the managed data
    StackAllocatorView device_view();

  private:
    // Number of bytes allocated
    size_type capacity_ = 0;
    // Stored memory on device
    DeviceAllocation allocation_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_StackAllocatorStore_hh
