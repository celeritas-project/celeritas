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
