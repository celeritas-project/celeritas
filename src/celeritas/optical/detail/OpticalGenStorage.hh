//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/StreamStore.hh"

#include "../OpticalGenData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct OpticalGenStorage
{
    using StoreT = StreamStore<OpticalGenParamsData, OpticalGenStateData>;

    StoreT obj;
    std::vector<OpticalBufferSize> size;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
