//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "detail/KernelTraitsImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Predicates used for \c __launch_bounds__ arguments
template<typename T>
inline constexpr bool kernel_no_bound
    = !detail::has_max_block_size_v<T> && !detail::has_min_warps_per_eu_v<T>;

template<typename T>
inline constexpr bool kernel_max_blocks
    = detail::has_max_block_size_v<T> && !detail::has_min_warps_per_eu_v<T>;

template<typename T>
inline constexpr bool kernel_max_blocks_min_warps
    = detail::has_max_block_size_v<T> && detail::has_min_warps_per_eu_v<T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
