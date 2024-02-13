//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngParams.cc
//---------------------------------------------------------------------------//
#include "CuHipRngParams.hh"

#include <utility>

#include "corecel/Assert.hh"

#include "CuHipRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a seed.
 */
CuHipRngParams::CuHipRngParams(unsigned int seed)
{
    HostVal<CuHipRngParamsData> host_data;
    host_data.seed = seed;
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<CuHipRngParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
