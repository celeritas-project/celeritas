//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    static_assert(sizeof(InitValueInt) == sizeof(int), "Bad size");

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
using OptionalInt = std::optional<int>;
struct Finalizer
{
    static OptionalInt& last_finalized()
    {
        static OptionalInt result;
        return result;
    }

    void operator()(int val) const { last_finalized() = val; }
};

OptionalInt get_last_finalized()
{
    return std::exchange(Finalizer::last_finalized(), {});
}

TEST(InitializedValue, finalizer)
{
    using InitValueInt = InitializedValue<int, Finalizer>;

    // Test default value
    InitValueInt{};
    EXPECT_EQ(std::nullopt, get_last_finalized());

    {
        // Test nondefault value
        InitValueInt derp{1};
        EXPECT_EQ(1, derp);
        EXPECT_EQ(std::nullopt, get_last_finalized());
    }
    EXPECT_EQ(1, get_last_finalized());

    InitValueInt ival{};
    EXPECT_EQ(0, ival);
    EXPECT_EQ(std::nullopt, get_last_finalized());

    {
        InitValueInt temp{2};
        ival = std::move(temp);
        EXPECT_EQ(std::nullopt, get_last_finalized());
        EXPECT_EQ(0, temp);
        EXPECT_EQ(2, ival);
    }
    EXPECT_EQ(std::nullopt, get_last_finalized());
    ival = {};
    EXPECT_EQ(2, get_last_finalized());

    InitValueInt other{345};
    EXPECT_EQ(345, other);
    ival = other;
    EXPECT_EQ(std::nullopt, get_last_finalized());
    EXPECT_EQ(345, ival);
    EXPECT_EQ(345, other);
    other = ival;
    EXPECT_EQ(345, get_last_finalized());
    EXPECT_EQ(345, ival);
    EXPECT_EQ(345, other);

    other = 1000;
    EXPECT_EQ(345, get_last_finalized());
    ival = std::move(other);
    EXPECT_EQ(345, get_last_finalized());
    EXPECT_EQ(1000, ival);
    EXPECT_EQ(0, other);

    InitValueInt third(std::move(ival));
    EXPECT_EQ(0, ival);
    EXPECT_EQ(1000, third);
    EXPECT_EQ(std::nullopt, get_last_finalized());

    // Test const T& constructor
    int const cint = 1234;
    third = InitValueInt(cint);
    EXPECT_EQ(1000, get_last_finalized());
    EXPECT_EQ(1234, third);

    // Test implicit conversion
    int tempint;
    tempint = third;
    EXPECT_EQ(1234, tempint);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
