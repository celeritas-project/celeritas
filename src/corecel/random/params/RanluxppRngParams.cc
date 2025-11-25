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

    // Compute seed_state (skip 2 ** 96 states)
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.state_2048, celeritas::RanluxppUInt(1) << 48);
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.seed_state, celeritas::RanluxppUInt(1) << 48);

    CELER_ASSERT(host_data);
    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
