//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepBuffer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/CollectionStateStore.hh"

#include "../StepData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Step storage shared across multiple actions.
 */
struct StepStorage
{
    template<MemSpace M>
    using StepStateCollection = CollectionStateStore<StepStateData, M>;
    template<MemSpace M>
    using MemSpaceTag = std::integral_constant<MemSpace, M>;

    // Parameter data
    CollectionMirror<StepParamsData> params;

    // State data
    struct
    {
        StepStateCollection<MemSpace::host>   host;
        StepStateCollection<MemSpace::device> device;
    } states;

    //!@{
    //! Tag-based dispatch for accessing states
    // TODO: replace with `if constexpr` for C++17
    StepStateCollection<MemSpace::host>& get_state(MemSpaceTag<MemSpace::host>)
    {
        return states.host;
    }

    StepStateCollection<MemSpace::device>&
    get_state(MemSpaceTag<MemSpace::device>)
    {
        return states.device;
    }
    //!@}
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
