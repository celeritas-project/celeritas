//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/InitializeRngState.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/InitializeRngState.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

// Create and initialize SplitMix64
celeritas::SplitMix64 makeSplitMix64(unsigned int seed,
                                     unsigned int event_id,
                                     unsigned int primary_id)
{
    celeritas::SplitMix64 rng(seed);
    rng.advance();
    rng.xor_state(event_id);
    rng.advance();
    rng.xor_state(primary_id);
    return rng;
}

//---------------------------------------------------------------------------//

TEST(InitializeRngStateTest, xorwow)
{
    unsigned int seed = 12345;
    unsigned int event_id = 5;
    unsigned int primary_id = 3;

    // Create a reference SplitMix64 engine
    SplitMix64 rng = makeSplitMix64(seed, event_id, primary_id);

    // Draw random numbers
    std::vector<std::uint64_t> vals(3);
    vals[0] = rng();
    rng.xor_state(event_id);
    vals[1] = rng();
    rng.xor_state(primary_id);
    vals[2] = rng();

    // Check that Xorwow initializer has the expected size
    XorwowRngEngine::RngStateInitializer_t ref_initializer;
    EXPECT_EQ(
        sizeof(std::uint32_t) * 6,
        sizeof(ref_initializer.xorstate[0]) * ref_initializer.xorstate.size()
            + sizeof(ref_initializer.weylstate));

    // Fill reference Xorwow initializer
    ref_initializer.xorstate[0] = static_cast<XorwowUInt>(vals[0]);
    ref_initializer.xorstate[1] = static_cast<XorwowUInt>(vals[0] >> 32);
    ref_initializer.xorstate[2] = static_cast<XorwowUInt>(vals[1]);
    ref_initializer.xorstate[3] = static_cast<XorwowUInt>(vals[1] >> 32);
    ref_initializer.xorstate[4] = static_cast<XorwowUInt>(vals[2]);
    ref_initializer.weylstate = static_cast<XorwowUInt>(vals[2] >> 32);

    // Create a test initializer
    XorwowSeed xorwow_seed;
    xorwow_seed[0] = seed;
    XorwowRngEngine::RngStateInitializer_t test_initializer;
    celeritas::initialize_rng_state(
        xorwow_seed, event_id, primary_id, test_initializer);

    EXPECT_VEC_EQ(ref_initializer.xorstate, test_initializer.xorstate);
    EXPECT_EQ(ref_initializer.weylstate, test_initializer.weylstate);
}

//---------------------------------------------------------------------------//

TEST(InitializeRngStateTest, ranluxpp)
{
    unsigned int seed = 12345;
    unsigned int event_id = 5;
    unsigned int primary_id = 3;

    // Create a reference SplitMix64 engine
    SplitMix64 rng = makeSplitMix64(seed, event_id, primary_id);

    // Draw 9 numbers and use them to fill the state
    celeritas::Array<std::uint64_t, 9> rng_vals;
    for (int i : celeritas::range(9))
    {
        rng_vals[i] = rng();
        if (i % 2 == 0)
        {
            rng.xor_state(event_id);
        }
        else
        {
            rng.xor_state(primary_id);
        }
    }

    // Create Ranluxpp initializer
    RanluxppRngEngine::RngStateInitializer_t ref_initializer;
    ref_initializer.value.number = rng_vals;

    // Check that Ranluxpp initializer has the expected size
    EXPECT_EQ(sizeof(std::uint64_t) * 9,
              sizeof(ref_initializer.value.number[0])
                  * ref_initializer.value.number.size());

    // Create a test initializer
    RanluxppRngEngine::RngStateInitializer_t test_initializer;
    celeritas::initialize_rng_state(
        seed, event_id, primary_id, test_initializer);

    EXPECT_VEC_EQ(ref_initializer.value.number, test_initializer.value.number);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
