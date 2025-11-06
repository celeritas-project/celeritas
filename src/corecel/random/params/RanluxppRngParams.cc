//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/params/RanluxppRngParams.cc
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#include "RanluxppRngParams.hh"

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
    host_data.max_position = 9 * 64;
    host_data.state_2048 = this->get_a_2048();

    // Compute a_seed, skipping 2 ** 96 states
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.state_2048, RanluxppUInt(1) << 48);
    host_data.seed_state = celeritas::detail::compute_power_modulus(
        host_data.seed_state, RanluxppUInt(1) << 48);

    CELER_ASSERT(host_data);
    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
/*!
 * Get the a polynomial.
 */
RanluxppArray9 const& RanluxppRngParams::get_a_2048() const
{
    static RanluxppArray9 const array_2048 = {
        0xed7faa90747aaad9,
        0x4cec2c78af55c101,
        0xe64dcb31c48228ec,
        0x6d8a15a13bee7cb0,
        0x20b2ca60cb78c509,
        0x256c3d3c662ea36c,
        0xff74e54107684ed2,
        0x492edfcc0cc8e753,
        0xb48c187cf5b22097,
    };
    return array_2048;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
