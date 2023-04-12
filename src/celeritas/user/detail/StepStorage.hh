//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepStorage.hh
//---------------------------------------------------------------------------//
#pragma once

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
    //// TYPES ////

    template<MemSpace M>
    using StepStateCollection = CollectionStateStore<StepStateData, M>;
    template<MemSpace M>
    using VecSSC = std::vector<StepStateCollection<M>>;

    //// DATA ////

    // Parameter data
    CollectionMirror<StepParamsData> params;

    // State data
    struct
    {
        VecSSC<MemSpace::host> host;
        VecSSC<MemSpace::device> device;
    } states;

    //// METHODS ////

    template<MemSpace M>
    decltype(auto) get_states()
    {
        if constexpr (M == MemSpace::host)
        {
            // NOTE: parens are necessary to return a reference instead of a
            // value
            return (states.host);
        }
        else
        {
            return (states.device);
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
