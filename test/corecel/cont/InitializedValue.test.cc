//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/InitializedValue.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/InitializedValue.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(InitializedValue, no_finalizer)
{
    using InitValueInt = InitializedValue<int>;
    // TODO: in C++20 use [[no_unique_address]] and restore this assertion
    // static_assert(sizeof(InitValueInt) == sizeof(int), "Bad size");

    // Use operator new to test that the int is being initialized properly by
    // constructing into data space that's been set to a different value
    alignas(int) std::byte buf[sizeof(InitValueInt)];
    std::fill(std::begin(buf), std::end(buf), std::byte(-1));
    InitValueInt* ival = new (buf) InitValueInt{};
    EXPECT_EQ(0, *ival);

    InitValueInt other = 345;
    EXPECT_EQ(345, other);
    *ival = other;
    EXPECT_EQ(345, *ival);
    EXPECT_EQ(345, other);
    other = 1000;
    *ival = std::move(other);
    EXPECT_EQ(1000, *ival);
    EXPECT_EQ(0, other);

    InitValueInt third(std::move(*ival));
    EXPECT_EQ(0, *ival);
    EXPECT_EQ(1000, third);

    // Test const T& constructor
    int const cint = 1234;
    other = InitValueInt(cint);
    EXPECT_EQ(1234, other);

    // Test implicit conversion
    int tempint;
    tempint = third;
    EXPECT_EQ(1000, tempint);
    tempint = 1;
#if 0
    // NOTE: this will not work because template matching will not
    // search for implicit constructors
    EXPECT_EQ(1000, std::max(tempint, third));
#else
    EXPECT_EQ(1000, std::max(tempint, static_cast<int>(third)));
#endif
    auto passthrough_int = [](int i) -> int { return i; };
    EXPECT_EQ(1000, passthrough_int(third));

    // Destroy
    ival->~InitializedValue();
}

//---------------------------------------------------------------------------//
struct Finalizer
{
    static std::vector<int>& finalized_values()
    {
        static std::vector<int> result;
        return result;
    }

    void operator()(int val) const { finalized_values().push_back(val); }
};

TEST(InitializedValue, finalizer)
{
    std::vector<int> expected;
    std::vector<int>& actual = Finalizer::finalized_values();
    actual.clear();

    using InitValueInt = InitializedValue<int, Finalizer>;

    InitValueInt ival{};
    EXPECT_EQ(0, ival);
    EXPECT_VEC_EQ(expected, actual);

    InitValueInt other{345};
    EXPECT_EQ(345, other);
    ival = other;
    expected.push_back(0);
    EXPECT_VEC_EQ(expected, actual);
    EXPECT_EQ(345, ival);
    EXPECT_EQ(345, other);

    other = 1000;
    expected.push_back(345);
    EXPECT_VEC_EQ(expected, actual);
    ival = std::move(other);
    expected.push_back(345);
    EXPECT_VEC_EQ(expected, actual);
    EXPECT_EQ(1000, ival);
    EXPECT_EQ(0, other);

    InitValueInt third(std::move(ival));
    EXPECT_EQ(0, ival);
    EXPECT_EQ(1000, third);

    // Test const T& constructor
    int const cint = 1234;
    other = InitValueInt(cint);
    expected.push_back(0);
    EXPECT_VEC_EQ(expected, actual);
    EXPECT_EQ(1234, other);

    // Test implicit conversion
    int tempint;
    tempint = third;
    EXPECT_EQ(1000, tempint);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
