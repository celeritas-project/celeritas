//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/EnumBitset.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/EnumBitset.hh"

#include <climits>

#include "corecel/Types.hh"
#include "corecel/cont/EnumBitsetIO.json.hh"
#include "corecel/io/EnumStringMapper.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
enum class Colors
{
    red,
    green,
    blue,
    size_
};

char const* to_cstring(Colors c)
{
    static EnumStringMapper<Colors> const get_impl{"red", "green", "blue"};

    return get_impl(c);
}

TEST(EnumJson, output)
{
    EnumBitset<Colors> c;

    {
        // Empty set
        nlohmann::json out = c;
        static char const expected[] = R"json([])json";
        EXPECT_JSON_EQ(expected, out.dump());
    }

    c[Colors::red] = true;
    c[Colors::blue] = true;

    {
        // Two values
        nlohmann::json out = c;
        static char const expected[] = R"json(["red","blue"])json";
        EXPECT_JSON_EQ(expected, out.dump());
    }
}

//---------------------------------------------------------------------------//

template<typename T>
class EnumBitsetTest : public Test
{
  protected:
    static constexpr size_type N = T::value;

    enum class MyEnum
    {
        // Normal enums would have values before this next one...
        antepenultimate = N - 3,
        penultimate,
        last,
        size_
    };

    static_assert(static_cast<int>(MyEnum::antepenultimate) >= 0
                  && static_cast<int>(MyEnum::size_) == N);
};

template<size_type N>
using Integral = std::integral_constant<size_type, N>;

using TestTypes = ::testing::Types<
   Integral<3>,
   Integral<16>,
   Integral<32>,
   Integral<48>,
   Integral<64>,
   Integral<65>,
   Integral<66>>;

TYPED_TEST_SUITE(EnumBitsetTest, TestTypes, );

TYPED_TEST(EnumBitsetTest, twiddling)
{
    using MyEnum = typename EnumBitsetTest<TypeParam>::MyEnum;
    using MyEnumBitset = EnumBitset<MyEnum>;
    using reference = typename MyEnumBitset::reference;

    constexpr auto N = TypeParam::value;
    constexpr auto last = MyEnum::last;
    constexpr auto penult = MyEnum::penultimate;

    MyEnumBitset x;
    EXPECT_TRUE(x.none());
    EXPECT_FALSE(x.any());
    EXPECT_FALSE(x.all());
    EXPECT_EQ(x.size(), N);

    x[last] = true;
    EXPECT_EQ(x.count(), 1);
    EXPECT_FALSE(x.none());
    EXPECT_TRUE(x.any());
    EXPECT_FALSE(x.all());
    EXPECT_TRUE(x[last]);

    x.flip(penult);
    EXPECT_TRUE(x[penult]);
    EXPECT_EQ(x.count(), 2);

    x.flip(penult);
    EXPECT_FALSE(x[penult]);

    x.flip(last);
    x.flip();
    EXPECT_TRUE(x.all());
    EXPECT_EQ(x.count(), N);

    x.reset();
    EXPECT_EQ(x.count(), 0);
    x.set(last);
    EXPECT_EQ(x.count(), 1);
    EXPECT_TRUE(x.any());
    EXPECT_TRUE(x[last]);

    x.reset();

    x.set();
    EXPECT_TRUE(x.all());
    EXPECT_EQ(x.count(), N);

    x.reset();

    x[penult] = true;
    EXPECT_TRUE(x[penult]);
    EXPECT_FALSE(~x[penult]);

    reference r = x[penult];
    bool b = x[penult];
    EXPECT_TRUE(r);
    EXPECT_TRUE(b);
    r = false;
    EXPECT_FALSE(x[penult]);
    EXPECT_FALSE(r);
    EXPECT_TRUE(b);

    x[penult] = ~x[penult];

    x[last] = x[penult];
    EXPECT_TRUE(x[last]);
    x[last].flip();
    EXPECT_FALSE(x[last]);
    x[last].flip();
    EXPECT_TRUE(x[last]);
    x[last] = x[last];
    EXPECT_TRUE(x[last]);
    x.reset(last);
    EXPECT_FALSE(x[last]);
}

TYPED_TEST(EnumBitsetTest, binary_ops)
{
    using MyEnum = typename EnumBitsetTest<TypeParam>::MyEnum;
    using MyEnumBitset = EnumBitset<MyEnum>;

    constexpr auto N = TypeParam::value;
    constexpr auto last = MyEnum::last;
    constexpr auto penult = MyEnum::penultimate;
    constexpr auto antepen = MyEnum::antepenultimate;

    MyEnumBitset x;
    x.flip(antepen);

    MyEnumBitset y;
    EXPECT_NE(x, y);
    y = x;
    EXPECT_EQ(x, y);

    x.reset();
    y.reset();
    x[last] = true;
    x[antepen] = true;
    y[penult] = true;
    y[antepen] = true;
    x ^= y;
    EXPECT_EQ(x.count(), 2);
    EXPECT_TRUE(x[penult]);
    EXPECT_FALSE(x[antepen]);

    x &= y;
    EXPECT_EQ(x.count(), 1);
    EXPECT_TRUE(x[penult]);

    x |= y;
    EXPECT_EQ(x.count(), 2);
    EXPECT_TRUE(x[penult]);
    EXPECT_TRUE(x[antepen]);

    x = ~x;
    EXPECT_EQ(x.count(), N - 2);
    EXPECT_TRUE(x[last]);
    EXPECT_FALSE(x[penult]);
    EXPECT_FALSE(x[antepen]);
}

TYPED_TEST(EnumBitsetTest, flag_init)
{
    using MyEnum = typename EnumBitsetTest<TypeParam>::MyEnum;
    using MyEnumBitset = EnumBitset<MyEnum>;
    using flags = typename MyEnumBitset::word_type;

    constexpr auto N = TypeParam::value;

    flags init{0};
    init = ~init;
    MyEnumBitset z(init);
    size_type bits_per_word = CHAR_BIT * sizeof(flags);
    EXPECT_EQ(z.count(), std::min(bits_per_word, N));
}

}  // namespace test
}  // namespace celeritas
