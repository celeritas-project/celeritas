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
 * Custom allocator for pinned host memory when Celeritas is built with device
 * support. Satisfies the Allocator named requirement.
 *
 * If Celeritas is built without device support, the allocator will fallback
 * to global \c ::operator new() / \c ::operator delete(). Only use this
 * when necessary (i.e. asynchronous H<->D memory transfer is needed)
 * as pinned memory reduces the memory available to the systems.
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
