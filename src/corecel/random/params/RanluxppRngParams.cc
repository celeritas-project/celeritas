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

    // Special multiplication constant used to advance the state by 1: equal to
    // the RCARRY 'a mod m' operator to the 2048 power
    host_data.advance_state = {
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

    // Multiplication constant to apply 2^96 'advance' operations
    host_data.advance_sequence
        = detail::compute_power_exp_modulus(host_data.advance_state, 96);

    CELER_ASSERT(host_data);
    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
