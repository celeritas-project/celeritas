//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OpaqueId.test.cc
//---------------------------------------------------------------------------//
#include "base/OpaqueId.hh"

#include <cstdint>
#include <utility>
#include "celeritas_test.hh"

using celeritas::OpaqueId;
struct TestInstantiator;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OpaqueIdTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(OpaqueIdTest, operations)
{
    using Id_t = OpaqueId<TestInstantiator, std::size_t>;

    Id_t unassigned;
    EXPECT_FALSE(unassigned);
    EXPECT_TRUE(!unassigned);
    EXPECT_EQ(unassigned, unassigned);
    EXPECT_EQ(unassigned, Id_t{});
#if CELERITAS_DEBUG
    EXPECT_THROW(unassigned.get(), celeritas::DebugError);
#endif
    EXPECT_EQ(static_cast<size_t>(-1), Id_t{}.unchecked_get());
    EXPECT_EQ(std::hash<std::size_t>()(-1), std::hash<Id_t>()(unassigned));

    Id_t assigned{123};
    EXPECT_TRUE(assigned);
    EXPECT_FALSE(!assigned);
    EXPECT_EQ(123, assigned.get());
    EXPECT_NE(unassigned, assigned);
    EXPECT_EQ(assigned, assigned);
    EXPECT_EQ(std::hash<std::size_t>()(123), std::hash<Id_t>()(assigned));

    EXPECT_EQ(10, Id_t{22} - Id_t{12});
    EXPECT_TRUE(Id_t{22} < Id_t{23});
}

TEST(OpaqueIdTest, multi_int)
{
    using UId8     = OpaqueId<TestInstantiator, std::uint_least8_t>;
    using Uint32   = std::uint_least32_t;
    using limits_t = std::numeric_limits<Uint32>;

    // Unassigned is always out-of-range
    EXPECT_FALSE(UId8{} < 0);
    EXPECT_FALSE(UId8{} < Uint32(limits_t::max()));
    EXPECT_FALSE(UId8{} < Uint32(255));
    EXPECT_FALSE(UId8{} < Uint32(256));
    EXPECT_FALSE(UId8{10} < Uint32(5));

    // Check 'true' conditions
    EXPECT_TRUE(UId8{254} < Uint32(limits_t::max()));
    EXPECT_TRUE(UId8{254} < Uint32(255));
    EXPECT_TRUE(UId8{10} < Uint32(15));
}
