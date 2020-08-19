//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocatorStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/DeviceVector.hh"
#include "SecondaryAllocatorPointers.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage device data for an allocation of secondaries.
 *
 * The capacity is known by the host, but the data and size are both stored on
 * device.
 */
class SecondaryAllocatorStore
{
  public:
    //@{
    //! Type aliases
    using size_type = SecondaryAllocatorPointers::size_type;
    //@}

  public:
    // Construct with no storage
    SecondaryAllocatorStore() = default;

    // Construct with the number of secondaries to allocate on device
    explicit SecondaryAllocatorStore(size_type capacity);

    // >>> HOST ACCESSORS

    //! Size of the allocation
    size_type capacity() const { return allocation_.size(); }

    // Get the actual size via a device->host copy
    size_type get_size();

    // Clear allocated data
    void clear();

    // >>> DEVICE ACCESSORS

    // Get a view to the managed data
    SecondaryAllocatorPointers device_pointers();

  private:
    DeviceVector<Secondary> allocation_;
    DeviceVector<size_type> size_allocation_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
