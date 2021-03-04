//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "DeviceVector.hh"
#include "StackAllocatorInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage device data for an allocation of a particular type.
 *
 * The capacity is known by the host, but the data and size are both stored on
 * device.
 */
template<class T>
class StackAllocatorStore
{
  public:
    //!@{
    //! Type aliases
    using value_type = T;
    using Pointers   = StackAllocatorPointers<T>;
    using size_type  = typename Pointers::size_type;
    //!@}

  public:
    // Construct with no storage
    StackAllocatorStore() = default;

    // Construct with the maximum number of values to store on device
    explicit StackAllocatorStore(size_type capacity);

    //// HOST ACCESSORS ////

    //! Size of the allocation
    size_type capacity() const { return allocation_.size(); }

    // Get the actual size via a device->host copy
    size_type get_size();

    // Clear allocated data (performs kernel launch!)
    void clear();

    //// DEVICE ACCESSORS ////

    // Get a view to the managed data
    Pointers device_pointers();

  private:
    DeviceVector<value_type> allocation_;
    DeviceVector<size_type>  size_allocation_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
