//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/InitializeRngState.hh
//---------------------------------------------------------------------------//
#pragma once

#include "RanluxppRngEngine.hh"
#include "SplitMix64.hh"
#include "XorwowRngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill a XorwowRngEngine state initializer given a seed, event id, and primary
 * id.
 */
CELER_FUNCTION void
initialize_rng_state(unsigned int seed,
                     unsigned int event_id,
                     unsigned int primary_id,
                     XorwowRngEngine::RngStateInitializer_t& rng_init)
{
    // Initialize SplitMix64 with the seed XORed with the track id
    SplitMix64 rng(seed ^ event_id ^ primary_id);

    // Fill first two state values
    std::uint64_t val = rng();
    rng_init.xorstate[0] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[1] = static_cast<XorwowUInt>(val >> 32);

    // XOR with event id
    rng.xor_state(event_id);
    val = rng();
    rng_init.xorstate[2] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[3] = static_cast<XorwowUInt>(val >> 32);

    // XOR with primary id
    rng.xor_state(primary_id);
    val = rng();
    rng_init.xorstate[4] = static_cast<XorwowUInt>(val);
    rng_init.weylstate = static_cast<XorwowUInt>(val >> 32);
}

//---------------------------------------------------------------------------//
/*!
 * Fill a Ranluxpp state initializer given a seed, event id, and primary id
 */
CELER_FUNCTION void
initialize_rng_state(unsigned int seed,
                     unsigned int event_id,
                     unsigned int primary_id,
                     RanluxppRngEngine::RngStateInitializer_t& rng_init)
{
    // Initialize SplitMix64 with the seed XORed with the track id
    SplitMix64 rng(seed ^ event_id ^ primary_id);

    // Fill first three state values
    rng_init.value.number[0] = rng();
    rng_init.value.number[1] = rng();
    rng_init.value.number[2] = rng();

    // XOR with event id and fill next three values
    rng.xor_state(event_id);
    rng_init.value.number[3] = rng();
    rng_init.value.number[4] = rng();
    rng_init.value.number[5] = rng();

    // XOR with step id and fill next three values
    rng.xor_state(primary_id);
    rng_init.value.number[6] = rng();
    rng_init.value.number[7] = rng();
    rng_init.value.number[8] = rng();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
