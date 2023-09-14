//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/PinnedAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Custom allocator for pinned host memory when Celeritas is built with device
 * support.
 *
 * Definition of the class is in a separate translation unit to hide CUDA/HIP
 * dependency from users, if support for a new type is needed, add an explicit
 * instantiation of the allocator for that type. If Celeritas is built without
 * device support, the allocator will fallback to \c std::malloc / \c std::free
 * Only use this when necessary (i.e. asynchronous H<->D memory transfer is
 * needed) as pinned memory reduces the memory available to the systems.
 */
template<class T>
struct PinnedAllocator
{
    using value_type = T;

    [[nodiscard]] T* allocate(std::size_t);

    void deallocate(T*, std::size_t) noexcept;
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
