//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/HitBuffer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Hit storage shared across multiple actions.
 */
struct HitBuffer
{
    template<MemSpace M>
    using HitStateCollection = CollectionStateStore<HitStateData, M>;

    // State data
    struct
    {
        HitStateCollection<MemSpace::host>   host;
        HitStateCollection<MemSpace::device> device;
    } states;

    //!@{
    //! Tag-based dispatch for accessing stores
    // TODO: replace with `if constexpr` for C++17
    HitStateCollection<MemSpace::host>
    get(std::integral_constant<MemSpace::host>)
    {
        return states.host;
    }

    HitStateCollection<MemSpace::device>
    get(std::integral_constant<MemSpace::device>)
    {
        return states.device;
    }
    //!@}
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
