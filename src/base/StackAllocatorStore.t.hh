//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorStore.cc
//---------------------------------------------------------------------------//
#include "StackAllocatorStore.hh"

#include "Assert.hh"
#include "Memory.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the number of bytes to allocate on device.
 */
template<class T>
StackAllocatorStore<T>::StackAllocatorStore(size_type capacity)
    : allocation_(capacity), size_allocation_(1)
{
    REQUIRE(capacity > 0);
    this->clear();
    ENSURE(this->get_size() == 0);
}

//---------------------------------------------------------------------------//
/*!
 * Get device pointers for the managed data.
 */
template<class T>
auto StackAllocatorStore<T>::device_pointers() -> Pointers
{
    REQUIRE(!allocation_.empty());
    Pointers ptrs;
    ptrs.storage = allocation_.device_pointers();
    ptrs.size    = size_allocation_.device_pointers().data();
    return ptrs;
}

//---------------------------------------------------------------------------//
/*!
 * Clear allocated data.
 *
 * This executes a kernel launch which simply resets the allocated size to
 * zero. It does not change the allocation itself.
 */
template<class T>
void StackAllocatorStore<T>::clear()
{
    REQUIRE(!size_allocation_.empty());
    device_memset_zero(size_allocation_.device_pointers());
}

//---------------------------------------------------------------------------//
/*!
 * Use a host->device copy to obtain the currently used size.
 *
 * This will have high latency because it's a host-device copy, and it should
 * \em definitely not be used if a kernel is potentially changing the size of
 * the stored data.
 */
template<class T>
auto StackAllocatorStore<T>::get_size() -> size_type
{
    REQUIRE(!size_allocation_.empty());
    size_type result;
    size_allocation_.copy_to_host({&result, 1});
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
