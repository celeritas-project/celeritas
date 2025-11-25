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
    host_data.state_2048 = {
        0xed7faa90747aaad9ull,
        0x4cec2c78af55c101ull,
        0xe64dcb31c48228ecull,
        0x6d8a15a13bee7cb0ull,
        0x20b2ca60cb78c509ull,
        0x256c3d3c662ea36cull,
        0xff74e54107684ed2ull,
        0x492edfcc0cc8e753ull,
        0xb48c187cf5b22097ull,
    };

    // Compute a_seed, skipping 2 ** 96 states
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.state_2048, RanluxppUInt(1) << 48);
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.seed_state, RanluxppUInt(1) << 48);

    CELER_ASSERT(host_data);
    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
