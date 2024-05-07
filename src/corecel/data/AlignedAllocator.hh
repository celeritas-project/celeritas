//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AlignedAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Allocate memory aligned to Align bytes. This can be useful if you want to overalign data, e.g., to make sure that allocations fall on a cache line boundary.
 * Alternatively, use alignas on your type declaration.
 */
template<class T, std::size_t Align = std::max(std::size_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}, alignof(T))>
struct AlignedAllocator
{
    static_assert(Align >= __STDCPP_DEFAULT_NEW_ALIGNMENT__, "Alignment must be at least as strict as __STDCPP_DEFAULT_NEW_ALIGNMENT__");
    static_assert(Align >= alignof(T), "Alignment must be at least as strict as alignof(T)");
    using value_type = T;

    template<class U>
    struct rebind
    {
        using other = AlignedAllocator<U, std::max(alignof(U), Align)>;
    };

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::bad_array_new_length();

        auto *p = ::operator new(n * sizeof(T), std::align_val_t{Align});
        if (!p)
        {
            throw std::bad_alloc();
        }
        CELER_ENSURE(reinterpret_cast<std::uintptr_t>(p) % Align == 0);
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept
    {
        ::operator delete(p, std::align_val_t{Align});
    }
};

template<class T, class U, std::size_t A1, std::size_t A2>
bool operator==(AlignedAllocator<T, A1> const&, AlignedAllocator<U, A2> const&)
{
    return A1 == A2;
}

template<class T, class U, std::size_t A1, std::size_t A2>
bool operator!=(AlignedAllocator<T, A1> const&, AlignedAllocator<U, A2> const&)
{
    return A1 != A2;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
