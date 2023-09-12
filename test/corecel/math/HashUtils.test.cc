//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/HashUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/HashUtils.hh"

#include <cstdint>

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(FnvHasherTest, four_byte)
{
    std::uint32_t result{0};
    auto hash = FnvHasher(&result);
    EXPECT_NE(0, result);
    hash(static_cast<Byte>(0xab));
    hash(static_cast<Byte>(0xcd));
    hash(static_cast<Byte>(0x19));
    EXPECT_EQ(0x111e8cf4u, result);
}

TEST(FnvHasherTest, eight_byte)
{
    std::uint64_t result{0};
    auto hash = FnvHasher(&result);
    EXPECT_NE(0, result);
    hash(static_cast<Byte>(0xab));
    hash(static_cast<Byte>(0xcd));
    hash(static_cast<Byte>(0x19));
    EXPECT_EQ(0x679fea1a6fe6ebb4ull, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//

TEST(HashUtilsTest, hash_combine)
{
    const std::string foo{"foo"};
    const std::string bar{"bar"};

    EXPECT_NE(hash_combine(), 0);
    EXPECT_NE(hash_combine(), hash_combine(0));
    EXPECT_NE(hash_combine(0, 1), hash_combine(1, 0));
    EXPECT_NE(hash_combine(foo, bar), hash_combine(bar, foo));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
