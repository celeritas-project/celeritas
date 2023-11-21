//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/PinnedAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Allocate pinned host memory when using a device.
 *
 * Satisfies the Allocator named requirement.
 *
 * If Celeritas is built without device support, or if the device is disabled
 * at runtime (including if this is called before the device is initialized!),
 * the allocator will fall back to global \code ::operator new() or \c
 * ::operator delete() \endcode. Only use this when necessary (i.e.
 * asynchronous H<->D memory transfer is needed) as pinned memory reduces the
 * memory available to the systems.
 */
template<class T>
struct PinnedAllocator
{
    using value_type = T;

    [[nodiscard]] T* allocate(std::size_t);

    void deallocate(T*, std::size_t) noexcept;
};

template<class T, class U>
bool operator==(PinnedAllocator<T> const&, PinnedAllocator<U> const&)
{
    return true;
}

template<class T, class U>
bool operator!=(PinnedAllocator<T> const&, PinnedAllocator<U> const&)
{
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
