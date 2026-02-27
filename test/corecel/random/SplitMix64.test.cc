//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/SplitMix64.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/engine/SplitMix64.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SplitMix64Test, draw_rng)
{
    celeritas::SplitMix64 sm(12345);

    EXPECT_EQ(2454886589211414944ul, sm());
    EXPECT_EQ(3778200017661327597ul, sm());
    EXPECT_EQ(2205171434679333405ul, sm());
    EXPECT_EQ(3248800117070709450ul, sm());
    EXPECT_EQ(9350289611492784363ul, sm());
}

TEST(SplitMix64Test, xor)
{
    std::uint64_t state1 = 12345;
    std::uint64_t state2 = 98765;

    // Create a test RNG for testing XOR function
    celeritas::SplitMix64 test_rng(state1);
    test_rng.xor_state(state2);

    // Create a pre-xored RNG
    celeritas::SplitMix64 ref_rng(state1 ^ state2);

    for ([[maybe_unused]] auto i : celeritas::range(10))
    {
        EXPECT_EQ(ref_rng(), test_rng());
    }
}

//---------------------------------------------------------------------------//

TEST(SplitMix64Test, advance)
{
    celeritas::SplitMix64 sm(12345);

    EXPECT_EQ(2454886589211414944ul, sm());
    sm.advance();
    EXPECT_EQ(2205171434679333405ul, sm());
    sm.advance();
    EXPECT_EQ(9350289611492784363ul, sm());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
