//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/InitializeRngState.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/InitializeRngState.hh"

#include "celeritas_test.hh"
// #include "InitializeRngState.test.hh"

namespace celeritas
{
namespace test
{

TEST(InitializeRngStateTest, xorwow)
{
    unsigned int seed = 12345;
    unsigned int event_id = 5;
    unsigned int primary_id = 3;

    // Create a reference SplitMix64 engine
    SplitMix64 rng(seed ^ event_id ^ primary_id);

    // Draw three numbers and use them to fill the state
    std::uint64_t val1 = rng();
    rng.xor_state(event_id);
    std::uint64_t val2 = rng();
    rng.xor_state(primary_id);
    std::uint64_t val3 = rng();

    // Check that Xorwow initializer has the expected size
    XorwowRngEngine::RngStateInitializer_t ref_initializer;
    EXPECT_EQ(
        sizeof(std::uint32_t) * 6,
        sizeof(ref_initializer.xorstate[0]) * ref_initializer.xorstate.size()
            + sizeof(ref_initializer.weylstate));

    // Create a reference Xorwow initializer
    ref_initializer.xorstate[0] = static_cast<XorwowUInt>(val1);
    ref_initializer.xorstate[1] = static_cast<XorwowUInt>(val1 >> 32);
    ref_initializer.xorstate[2] = static_cast<XorwowUInt>(val2);
    ref_initializer.xorstate[3] = static_cast<XorwowUInt>(val2 >> 32);
    ref_initializer.xorstate[4] = static_cast<XorwowUInt>(val3);
    ref_initializer.weylstate = static_cast<XorwowUInt>(val3 >> 32);

    // Create a test initializer
    XorwowRngEngine::RngStateInitializer_t test_initializer;
    celeritas::initialize_rng_state(
        seed, event_id, primary_id, test_initializer);

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
    SplitMix64 rng(seed ^ event_id ^ primary_id);

    // Draw 9 numbers and use them to fill the state
    celeritas::Array<std::uint64_t, 9> rng_vals;
    rng_vals[0] = rng();
    rng_vals[1] = rng();
    rng_vals[2] = rng();
    rng.xor_state(event_id);
    rng_vals[3] = rng();
    rng_vals[4] = rng();
    rng_vals[5] = rng();
    rng.xor_state(primary_id);
    rng_vals[6] = rng();
    rng_vals[7] = rng();
    rng_vals[8] = rng();

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
