//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/HashUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/HashUtils.hh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct PaddedStruct
{
    bool b;
    int i;
    long long int lli;
};
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

namespace std
{
//---------------------------------------------------------------------------//
template<>
struct hash<celeritas::test::PaddedStruct>
{
    std::size_t operator()(celeritas::test::PaddedStruct const& s) const
    {
        return celeritas::hash_combine(s.b, s.i, s.lli);
    }
};
//---------------------------------------------------------------------------//
}  // namespace std

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
    hash(static_cast<std::byte>(0xab));
    hash(static_cast<std::byte>(0xcd));
    hash(static_cast<std::byte>(0x19));
    EXPECT_EQ(0x111e8cf4u, result);
}

TEST(FnvHasherTest, eight_byte)
{
    std::uint64_t result{0};
    auto hash = FnvHasher(&result);
    EXPECT_NE(0, result);
    hash(static_cast<std::byte>(0xab));
    hash(static_cast<std::byte>(0xcd));
    hash(static_cast<std::byte>(0x19));
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
    std::string const foo{"foo"};
    std::string const bar{"bar"};

    EXPECT_NE(hash_combine(), 0);
    EXPECT_NE(hash_combine(), hash_combine(0));
    EXPECT_NE(hash_combine(0, 1), hash_combine(1, 0));
    EXPECT_NE(hash_combine(foo, bar), hash_combine(bar, foo));
}

//---------------------------------------------------------------------------//
struct UnpaddedStruct
{
    int i;
    int j;
};

TEST(HashSpan, padded_struct)
{
    PaddedStruct temp;
    std::memset(&temp, 0x0f, sizeof(temp));
    temp.b = false;
    temp.i = 0x1234567;
    temp.lli = 0xabcde01234ll;
    Span<PaddedStruct const, 1> s{&temp, 1};
    EXPECT_EQ(std::hash<decltype(s)>{}(s),
              hash_combine(hash_combine(temp.b, temp.i, temp.lli)));
}

TEST(HashSpan, unpadded_struct)
{
    static int const values[] = {0x1234567, 0x2345678};
    UnpaddedStruct temp;
    temp.i = values[0];
    temp.j = values[1];
    Span<UnpaddedStruct const, 1> s{&temp, 1};
    EXPECT_EQ(std::hash<decltype(s)>{}(s), hash_as_bytes(s));
}

TEST(HashSpan, reals)
{
    if (CELERITAS_DEBUG)
    {
        auto nan = std::numeric_limits<double>::quiet_NaN();
        Span<double const, 1> s{&nan, 1};
        EXPECT_THROW(std::hash<decltype(s)>{}(s), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
