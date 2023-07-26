
//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ExecutorInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

namespace celeritas
{

template<typename T, typename = bool>
struct has_max_block_size : std::false_type
{
};

template<typename T>
struct has_max_block_size<T, decltype((void)T::max_block_size, true)>
    : std::true_type
{
};

template<typename T>
constexpr bool has_max_block_size_v = has_max_block_size<T>::value;

template<typename T, typename = bool>
struct has_min_warps_per_eu : std::false_type
{
};

template<typename T>
struct has_min_warps_per_eu<T, decltype((void)T::min_warps_per_eu, true)>
    : std::true_type
{
};

template<typename T>
constexpr bool has_min_warps_per_eu_v = has_min_warps_per_eu<T>::value;
//---------------------------------------------------------------------------//
}  // namespace celeritas