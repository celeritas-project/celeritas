//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/GenStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/StreamStore.hh"

#include "../OpticalGenData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct GenStorage
{
    using StoreT = StreamStore<OpticalGenParamsData, OpticalGenStateData>;

    StoreT obj;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas