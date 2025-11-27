//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/RanluxppRngParams.cc
//---------------------------------------------------------------------------//
#include "RanluxppRngParams.hh"

#include "corecel/random/engine/detail/RanluxppImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a seed.
 */
RanluxppRngParams::RanluxppRngParams(RanluxppUInt seed)
{
    HostVal<RanluxppRngParamsData> host_data;

    // Save basic data
    host_data.seed = seed;

    CELER_ASSERT(host_data);
    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
