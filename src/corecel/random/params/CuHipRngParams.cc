//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/CuHipRngParams.cc
//---------------------------------------------------------------------------//
// This need to be included before Stream.hh
// because it defines the stream type for CUDA/HIP
#include "corecel/DeviceRuntimeApi.hh"

// Main header include
#include "CuHipRngParams.hh"

// Headers needed for implementation
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
