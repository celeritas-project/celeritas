//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/CuHipRngParams.cc
//---------------------------------------------------------------------------//
#include "CuHipRngParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/random/data/CuHipRngData.hh"

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
    data_ = ParamsDataStore<CuHipRngParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
