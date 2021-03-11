//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Range.test.cc
//---------------------------------------------------------------------------//
#include "base/Range.hh"

#include "celeritas_test.hh"
#include "Range.test.hh"

using namespace celeritas_test;

using celeritas::count;
using celeritas::range;

using VecInt   = std::vector<int>;
using Vec_UInt = std::vector<unsigned int>;

enum class Color : unsigned int
{
    red,
    green,
    blue,
    size_ //!< Note: "size_" is necessary to take a range of enums
};

enum class WontWorkColors
{
    red   = 1,
    green = 2,
    blue  = 4,
};

namespace pokemon
{
enum Pokemon
{
    charmander = 0,
    bulbasaur,
    squirtle,
    pikachu,
    size_
};
}

namespace fake_geant
{
enum G4MySillyIndex
{
    DoIt = 0,
    ooOO00OOoo,
    PostStepGetPhysicalInteractionLength,
    NumberOfGeantIndex
};
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST(RangeTest, class_interface)
{
    using RangeT = celeritas::Range<int>;
    RangeT r(1, 3);
    EXPECT_EQ(2, r.size());
    EXPECT_EQ(1, *r.begin());
    EXPECT_EQ(3, *r.end());
    EXPECT_EQ(1, r.front());
    EXPECT_EQ(2, r.back());
    EXPECT_FALSE(r.empty());

    r = RangeT(5);
    EXPECT_EQ(5, r.size());
    EXPECT_EQ(0, r.front());
    EXPECT_EQ(4, r.back());

    r = RangeT();
    EXPECT_EQ(0, r.size());
    EXPECT_TRUE(r.empty());
}

TEST(RangeTest, ints)
{
    VecInt vals;
    for (auto i : range(0, 4))
    {
        ASSERT_EQ(sizeof(int), sizeof(i));
        vals.push_back(i);
    }
    EXPECT_VEC_EQ((VecInt{0, 1, 2, 3}), vals);
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

    EXPECT_VEC_EQ((Vec_UInt{20, 22, 24}), vals);
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
    EXPECT_VEC_EQ((VecInt{0, 1, 2, 3}), vals);
}

TEST(RangeTest, vec_fill)
{
    auto r = celeritas::range(1, 5);
    EXPECT_EQ(5 - 1, r.size());
    EXPECT_FALSE(r.empty());

    VecInt vals(r.begin(), r.end());

    EXPECT_VEC_EQ((VecInt{1, 2, 3, 4}), vals);
    EXPECT_EQ(5 - 1, r.size());
    EXPECT_FALSE(r.empty());

    // Re-assign
    vals.assign(r.begin(), r.end());
    EXPECT_VEC_EQ((VecInt{1, 2, 3, 4}), vals);

    r = celeritas::range(5, 5);
    EXPECT_EQ(0, r.size());
    EXPECT_TRUE(r.empty());
}

TEST(RangeTest, empty)
{
    celeritas::Range<int> r;
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(0, r.size());
}

TEST(RangeTest, enums)
{
    int  ctr          = 0;
    auto most_pokemon = celeritas::range(pokemon::charmander, pokemon::size_);
    for (pokemon::Pokemon p : most_pokemon)
    {
        EXPECT_EQ(p, static_cast<pokemon::Pokemon>(ctr));
        ++ctr;
    }
    // Size of a range that returns enums
    EXPECT_EQ(4, most_pokemon.size());

    ctr = 0;
    for (auto p : celeritas::range(pokemon::size_))
    {
        static_assert(std::is_same<decltype(p), pokemon::Pokemon>::value,
                      "Pokemon range should be an enum");
        EXPECT_EQ(p, static_cast<pokemon::Pokemon>(ctr));
        ++ctr;
    }

#if CELERITAS_DEBUG
    EXPECT_THROW(celeritas::range(static_cast<pokemon::Pokemon>(100)),
                 celeritas::DebugError);
#endif
}

TEST(RangeTest, different_enums)
{
    int ctr = 0;
    for (auto i : celeritas::range(fake_geant::NumberOfGeantIndex))
    {
        static_assert(
            std::is_same<decltype(i), fake_geant::G4MySillyIndex>::value,
            "G4 range should be an enum");
        EXPECT_EQ(i, static_cast<fake_geant::G4MySillyIndex>(ctr));
        ++ctr;
    }
}

TEST(RangeTest, enum_step)
{
    EXPECT_TRUE((std::is_same<std::underlying_type<pokemon::Pokemon>::type,
                              unsigned int>::value));

    /*!
     * The following should fail to compile because enums cannot be added to
     * ints.
     */
    std::vector<int> vals;
    for (auto p : celeritas::range(pokemon::size_).step(3u))
    {
        static_assert(std::is_same<decltype(p), pokemon::Pokemon>::value,
                      "Pokemon range should still be an enum");
        vals.push_back(static_cast<int>(p));
    }
    EXPECT_VEC_EQ((VecInt{0, 3}), vals);
}

TEST(RangeTest, enum_classes)
{
    int ctr = 0;
    for (Color c : celeritas::range(Color::red, Color::size_))
    {
        EXPECT_EQ(c, static_cast<Color>(ctr));
        ++ctr;
    }

    auto srange = celeritas::range(Color::red, Color::size_).step(2u);
    EXPECT_EQ(Color::red, *srange.begin());
    EXPECT_EQ(static_cast<int>(Color::size_), static_cast<int>(*srange.end()));
    EXPECT_EQ(srange.begin(), srange.begin());
    EXPECT_NE(srange.begin(), srange.end());

    std::vector<int> vals;
    for (auto c : srange)
    {
        static_assert(std::is_same<decltype(c), Color>::value,
                      "Color range should still be an enum");
        vals.push_back(static_cast<int>(c));
    }
    EXPECT_VEC_EQ((VecInt{0, 2}), vals);

    /*!
     * The following should fail because the enum doesn't have a size_ member.
     *
     * \verbatim
     * ../src/base/detail/RangeImpl.hh:61:24: error: no member named 'size_' in
     * 'WontWorkColors' \endverbatim
     */
#if 0
    range(WontWorkColors::blue);
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
    EXPECT_VEC_EQ((VecInt{4, 3, 2, 1, 0}), vals);

    vals.clear();
    for (auto i : range(6).step(-2))
    {
        vals.push_back(i);
        if (i > 7 || i < -3)
            break;
    }
    EXPECT_VEC_EQ((VecInt{4, 2, 0}), vals);
}

TEST(RangeTest, backward_conversion)
{
    VecInt vals;

    /*!
     * The following should fail to compile: "non-constant-expression cannot be
     * narrowed from type 'short' to 'unsigned int' in initializer list"
     * rightly showing that you can't step an unsigned int backward with the
     * current implementation of range.
     */
#if 0
    range<unsigned int>(5).step<signed short>(-1);
#endif

    // Result of 'step' should be original range type
    for (auto i : range<int>(5).step<signed short>(-1))
    {
        static_assert(std::is_same<decltype(i), int>::value,
                      "Range result should be converted to int!");
        vals.push_back(i);
        if (i > 7 || i < -2)
            break;
    }
    EXPECT_VEC_EQ((VecInt{4, 3, 2, 1, 0}), vals);
}

TEST(RangeTest, opaque_id)
{
    using MatId = celeritas::OpaqueId<struct Mat>;

    {
        celeritas::Range<MatId> fr;
        EXPECT_EQ(0, fr.size());
        EXPECT_TRUE(fr.empty());
    }
    {
        celeritas::Range<MatId> r(MatId{10});
        EXPECT_EQ(10, r.size());
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(MatId{0}, *r.begin());
        EXPECT_EQ(MatId{3}, r[3]);
        EXPECT_EQ(MatId{10}, *r.end());
    }
    {
        celeritas::Range<MatId> r(MatId{3}, MatId{10});
        EXPECT_EQ(7, r.size());
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(MatId{3}, *r.begin());
        EXPECT_EQ(MatId{5}, r[2]);
        EXPECT_EQ(MatId{10}, *r.end());
    }

    VecInt vals;
    for (auto id : range(MatId{4}, MatId{6}))
    {
        vals.push_back(id.unchecked_get());
    }
    EXPECT_VEC_EQ((VecInt{4, 5}), vals);

    auto srange = range(MatId{6}).step(3u);
    EXPECT_EQ(MatId{0}, *srange.begin());
    EXPECT_EQ(MatId{6}, *srange.end());

    EXPECT_EQ(4, range(MatId{4}).size());
}

TEST(CountTest, infinite)
{
    VecInt vals;

    auto counter = count<int>().begin();
    vals.push_back(*counter++);
    vals.push_back(*counter++);
    vals.push_back(*counter++);

    EXPECT_VEC_EQ((VecInt{0, 1, 2}), vals);

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
    EXPECT_VEC_EQ((VecInt{10, 25, 40, 55, 70, 85}), vals);

    EXPECT_EQ(10, *celeritas::Count<int>(10).begin());
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

    EXPECT_VEC_EQ((VecInt{3, 2, 1, 0, -1}), vals);
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
TEST(TEST_IF_CELERITAS_CUDA(DeviceRangeTest), grid_stride)
{
    // next prime after 1<<20 elements to avoid multiples of block/stride
    unsigned int N = 1048583;

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
