//---------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorStore.cc
//---------------------------------------------------------------------------//
#include "StackAllocatorStore.hh"

#include "Assert.hh"
#include "Macros.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the number of bytes to allocate on device.
 */
StackAllocatorStore::StackAllocatorStore(size_type capacity)
    : capacity_(capacity), allocation_(capacity + sizeof(size_type))
{
    REQUIRE(capacity > 0 && capacity % 16 == 0);

    // Reset the stored values to zero
    this->clear();

    ENSURE(allocation_.size() > this->capacity());
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
StackAllocatorView StackAllocatorStore::device_view()
{
    StackAllocatorView view;
    view.storage = span<byte>{allocation_.device_view().data(), capacity_};
    view.size = reinterpret_cast<size_type*>(view.storage.data() + capacity_);
    return view;
}

//---------------------------------------------------------------------------//
/*!
 * Clear allocated data.
 */
void StackAllocatorStore::clear()
{
    detail::device_memset_zero(allocation_.device_view());
}

//---------------------------------------------------------------------------//
/*!
 * Swap with another stack allocator.
 */
void StackAllocatorStore::swap(StackAllocatorStore& other)
{
    using std::swap;
    swap(capacity_, other.capacity_);
    swap(allocation_, other.allocation_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
