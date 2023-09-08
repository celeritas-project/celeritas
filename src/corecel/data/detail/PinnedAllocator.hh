//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/PinnedAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>
#include <limits>
#include <new>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/device_runtime_api.h"

namespace celeritas
{
namespace detail
{
template<class T>
struct PinnedAllocator
{
    using value_type = T;

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        void* p{nullptr};
        CELER_DEVICE_CALL_PREFIX(MallocHost(&p, n * sizeof(T)));
        if(p)
        {
            return static_cast<T*>(p);
        }
        throw std::bad_alloc();
    }

    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept
    {
        //Not using CELER_DEVICE_CALL_PREFIX, must be noexcept
        CELER_DEVICE_PREFIX(FreeHost(p));
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
