//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/StreamStore.hh"

#include "../StepData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct StepStorage
{
    using StoreT = StreamStore<StepParamsData, StepStateData>;

    StoreT obj;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
