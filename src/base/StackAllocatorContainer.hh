//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorContainer.hh
//---------------------------------------------------------------------------//
#ifndef base_StackAllocatorContainer_hh
#define base_StackAllocatorContainer_hh

#include <memory>
#include "Types.hh"
#include "StackAllocatorView.hh"

namespace celeritas
{
struct StackAllocatorContainerPimpl;
//---------------------------------------------------------------------------//
/*!
 * Manage a chunk of device memory as an in-kernel stack.
 *
 * This low-level class should be used by other containers to ensure alignment
 * requirements are met.
 */
class StackAllocatorContainer
{
  public:
    //@{
    //! Public types
    using size_type = StackAllocatorView::size_type;
    //@}
  public:
    // Construct with the number of bytes to allocate on device
    explicit StackAllocatorContainer(size_type capacity);

    //@{
    //! Defaults that cause thrust to launch kernels
    StackAllocatorContainer();
    ~StackAllocatorContainer();
    StackAllocatorContainer(StackAllocatorContainer&&);
    StackAllocatorContainer& operator=(StackAllocatorContainer&&);
    //@}

    // >>> HOST ACCESSORS

    //! Size (in bytes) of the allocation
    size_type capacity() const { return capacity_; }

    // Clear allocated data
    void clear();

    // Swap with another stack allocator
    void swap(StackAllocatorContainer& other);

    // >>> DEVICE ACCESSORS

    // Get a view to the managed data
    StackAllocatorView device_view() const;

  private:
    // Number of bytes allocated
    size_type capacity_ = 0;
    // Stored memory on device
    std::unique_ptr<StackAllocatorContainerPimpl> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_StackAllocatorContainer_hh
