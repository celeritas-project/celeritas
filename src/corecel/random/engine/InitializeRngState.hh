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
CELER_FUNCTION inline void
initialize_rng_state(XorwowSeed seed,
                     unsigned int event_id,
                     unsigned int primary_id,
                     XorwowRngEngine::RngStateInitializer_t& rng_init)
{
    // Seed and setup SplitMix64 with the seed, event_id, and primary_id
    // incorporated into the state
    SplitMix64 rng(seed[0]);
    for (unsigned int v : {event_id, primary_id})
    {
        rng.advance();
        rng.xor_state(v);
    }

    // Fill initial two states
    std::uint64_t val = rng();
    rng_init.xorstate[0] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[1] = static_cast<XorwowUInt>(val >> 32);

    // Update SplitMix64 state by xor-ing in event_id and draw another number
    // for the next two state values
    rng.xor_state(event_id);
    val = rng();
    rng_init.xorstate[2] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[3] = static_cast<XorwowUInt>(val >> 32);

    // Update SplitMix64 state by xor-ing in primary id and update the final
    // two numbers
    rng.xor_state(primary_id);
    val = rng();
    rng_init.xorstate[4] = static_cast<XorwowUInt>(val);
    rng_init.weylstate = static_cast<XorwowUInt>(val >> 32);
}

//---------------------------------------------------------------------------//
/*!
 * Fill a Ranluxpp state initializer given a seed, event id, and primary id
 */
CELER_FUNCTION inline void
initialize_rng_state(RanluxppUInt seed,
                     unsigned int event_id,
                     unsigned int primary_id,
                     RanluxppRngEngine::RngStateInitializer_t& rng_init)
{
    // Seed and setup SplitMix64 with the seed, event_id, and primary_id
    // incorporated into the state
    SplitMix64 rng(seed);
    for (unsigned int v : {event_id, primary_id})
    {
        rng.advance();
        rng.xor_state(v);
    }

    // Loop and fill the values in the Ranluxpp state
    for (unsigned int i : celeritas::range(9))
    {
        rng_init.value.number[i] = rng();
        rng.xor_state(i % 2 == 0 ? event_id : primary_id);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
