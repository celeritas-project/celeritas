//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Range.test.cc
//---------------------------------------------------------------------------//
#include "base/Range.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "Range.test.hh"

using namespace celeritas_test;

using celeritas::count;
using celeritas::range;

using VecInt   = std::vector<int>;
using Vec_UInt = std::vector<unsigned int>;

enum class Colors : unsigned int
{
    RED = 0,
    GREEN,
    BLUE,
    YELLOW,
    END_COLORS
};

enum Pokemon
{
    CHARMANDER = 0,
    BULBASAUR,
    SQUIRTLE,
    PIKACHU,
    END_POKEMON
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(RangeTest, ints)
{
    VecInt vals;
    for (auto i : range(0, 4))
    {
        ASSERT_EQ(sizeof(int), sizeof(i));
        vals.push_back(i);
    }
    // EXPECT_VEC_EQ((VecInt{0,1,2,3}), vals);
}

TEST(RangeTest, chars)
{
    for (auto i : range('A', 'Z'))
    {
        cout << i;
    }
    cout << endl;
}

TEST(RangeTest, uint)
{
    Vec_UInt vals;

    for (auto u : range(20u, 25u).step(2u))
    {
        vals.push_back(u);
    }

    // EXPECT_VEC_EQ((Vec_UInt{20,22,24}), vals);
}

TEST(RangeTest, large)
{
    using large_int = std::uint_least64_t;

    // Note: you can't pass both 0 (int) and large_int(10) , because the range
    // can't figure out T
    for (auto i : range<large_int>(0, 10))
    {
        ASSERT_EQ(sizeof(large_int), sizeof(i));
    }
}

TEST(RangeTest, just_end)
{
    VecInt vals;

    for (auto i : range(4))
    {
        vals.push_back(i);
        ASSERT_LT(i, 10);
    }
    // EXPECT_VEC_EQ((VecInt{0,1,2,3}), vals);
}

TEST(RangeTest, vec_fill)
{
    auto r = celeritas::range(1, 5);
    EXPECT_EQ(5 - 1, r.size());
    EXPECT_FALSE(r.empty());

    VecInt vals(r.begin(), r.end());

    // EXPECT_VEC_EQ((VecInt{1,2,3,4}), vals);
    EXPECT_EQ(5 - 1, r.size());
    EXPECT_FALSE(r.empty());

    // Re-assign
    vals.assign(r.begin(), r.end());
    // EXPECT_VEC_EQ((VecInt{1,2,3,4}), vals);

    r = celeritas::range(5, 5);
    EXPECT_EQ(0, r.size());
    EXPECT_TRUE(r.empty());
}

TEST(RangeTest, empty)
{
    celeritas::FiniteRange<int> r;
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(0, r.size());
}

TEST(RangeTest, enums)
{
    int  ctr          = 0;
    auto most_pokemon = celeritas::range(CHARMANDER, END_POKEMON);
    for (Pokemon p : most_pokemon)
    {
        EXPECT_EQ(p, (Pokemon)ctr);
        ++ctr;
    }
    // Size of a range that returns enums
    EXPECT_EQ(4, most_pokemon.size());

    ctr = 0;
    for (Pokemon p : celeritas::range(END_POKEMON))
    {
        EXPECT_EQ(p, (Pokemon)ctr);
        ++ctr;
    }
}

TEST(RangeTest, enum_step)
{
    // Since the result of enum + int is int, this is OK --
    int ctr = 0;
    for (auto p : celeritas::range(END_POKEMON).step(2))
    {
        static_assert(std::is_same<decltype(p), int>::value,
                      "Range result should be converted to int!");
        EXPECT_EQ(p, ctr);
        ctr += 2;
    }
}

TEST(RangeTest, enum_classes)
{
    int ctr = 0;
    for (Colors c : celeritas::range(Colors::RED, Colors::END_COLORS))
    {
        EXPECT_EQ(c, (Colors)ctr);
        ++ctr;
    }

    /*!
     * The following should fail to compile because there's no common type
     * between int and enum class.
     */
#if 0
    celeritas::range(Colors::END_COLORS).step(1);
#endif
}

TEST(RangeTest, backward)
{
    VecInt vals;

    for (auto i : range(5).step(-1))
    {
        vals.push_back(i);
        if (i > 6 || i < -1)
            break;
    }
    // EXPECT_VEC_EQ((VecInt{4, 3, 2, 1, 0}), vals);

    vals.clear();
    for (auto i : range(6).step(-2))
    {
        vals.push_back(i);
        if (i > 7 || i < -3)
            break;
    }
    // EXPECT_VEC_EQ((VecInt{4, 2, 0}), vals);
}

TEST(RangeTest, backward_conversion)
{
    VecInt vals;

    /*!
     * Note that the static assertion evaluates to false because there is no
     * integer type that encompasses the full range of both int and unsigned
     * long
     * https://stackoverflow.com/questions/15211463/why-isnt-common-typelong-unsigned-longtype-long-long
     */
#if 0
    static_assert(
        std::is_same<typename std::common_type<int, unsigned long long>::type,
                     long long>::value,
        "Integer conversions are weird!");
#endif

    /*!
     * The following should raise an error: "non-constant-expression cannot be
     * narrowed from type 'short' to 'unsigned int' in initializer list"
     * rightly showing that you can't step an unsigned int backward with the
     * current implementation of range.
     */
#if 0
    range<unsigned int>(5).step<signed short>(-1);
#endif

    // Result of 'step' should be common type of ULL and int
    for (auto i : range<int>(5).step<signed short>(-1))
    {
        static_assert(std::is_same<decltype(i), int>::value,
                      "Range result should be converted to int!");
        vals.push_back(i);
        if (i > 7 || i < -2)
            break;
    }
    // EXPECT_VEC_EQ((VecInt{4, 3, 2, 1, 0}), vals);
}

TEST(CountTest, infinite)
{
    VecInt vals;

    auto counter = count<int>().begin();
    vals.push_back(*counter++);
    vals.push_back(*counter++);
    vals.push_back(*counter++);

    // EXPECT_VEC_EQ((VecInt{0, 1, 2}), vals);

    EXPECT_FALSE(count<int>().empty());
}

TEST(CountTest, start)
{
    VecInt vals;

    for (auto i : count(10).step(15))
    {
        if (i > 90)
            break;
        vals.push_back(i);
    }
    // EXPECT_VEC_EQ((VecInt{10, 25, 40, 55, 70, 85}), vals);
}

TEST(CountTest, backward)
{
    VecInt vals;

    // Count backward from 3 to -ininity
    for (auto i : count(3).step(-1))
    {
        if (i > 5 || i < -1)
            break;
        vals.push_back(i);
    }

    // EXPECT_VEC_EQ((VecInt{3, 2, 1, 0, -1}), vals);
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
#if CELERITAS_USE_CUDA
TEST(DeviceRangeTest, grid_stride)
{
    // ~1M elements for saxpy
    unsigned int N = 1 << 20;

    // Set Inputs
    RangeTestInput input;
    input.a = 3;
    input.x.assign(N, 1);
    // y varies with index so we can verify common order on CPU vs Device
    input.y.assign(N, 0);
    for (auto i : range(N))
    {
        input.y[i] = i;
    }
    input.num_threads = 32;
    input.num_blocks  = 256;

    // Calculate saxpy using CPU
    std::vector<int> z_cpu(N, 0.0);
    for (auto i : range(N))
    {
        z_cpu[i] = input.a * input.x[i] + input.y[i];
    }

    // Calculate saxpy on Device
    RangeTestOutput result = rangedev_test(input);
    EXPECT_VEC_EQ(z_cpu, result.z);
}

#endif