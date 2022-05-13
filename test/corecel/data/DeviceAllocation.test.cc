//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAllocation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/DeviceAllocation.hh"

#include <algorithm>

#include "corecel/cont/Span.hh"

#include "celeritas_test.hh"

using celeritas::Byte;
using celeritas::DeviceAllocation;

TEST(InitializedValue, semantics)
{
    using InitValueInt = celeritas::detail::InitializedValue<int>;
    static_assert(sizeof(InitValueInt) == sizeof(int), "Bad size");

    // Use operator new to test that the int is being initialized properly by
    // constructing into data space that's been set to a different value
    alignas(int) Byte buf[sizeof(int)];
    std::fill(std::begin(buf), std::end(buf), Byte(-1));
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
    const int cint = 1234;
    other          = InitValueInt(cint);
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
// TEST HARNESS
//---------------------------------------------------------------------------//

class DeviceAllocationTest : public celeritas_test::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DeviceAllocationTest, always)
{
    DeviceAllocation alloc;
    EXPECT_EQ(0, alloc.size());
    EXPECT_TRUE(alloc.empty());
}

#if !CELER_USE_DEVICE
TEST_F(DeviceAllocationTest, nocuda)
{
    // Can't allocate
    EXPECT_THROW(DeviceAllocation(1234), celeritas::DebugError);
}
#endif

TEST_F(DeviceAllocationTest, TEST_IF_CELER_DEVICE(device))
{
    DeviceAllocation alloc(1024);
    EXPECT_EQ(1024, alloc.size());
    EXPECT_FALSE(alloc.empty());

    {
        DeviceAllocation other(128);
        Byte*            orig_alloc = alloc.device_ref().data();
        Byte*            orig_other = other.device_ref().data();
        swap(alloc, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, alloc.device_ref().data());
        EXPECT_EQ(orig_alloc, other.device_ref().data());
    }
    EXPECT_EQ(128, alloc.size());

    std::vector<Byte> data(alloc.size());
    data.front() = Byte(1);
    data.back()  = Byte(127);

    alloc.copy_to_device(celeritas::make_span(data));

    std::vector<Byte> newdata(alloc.size());
    alloc.copy_to_host(celeritas::make_span(newdata));
    EXPECT_EQ(Byte(1), newdata.front());
    EXPECT_EQ(Byte(127), newdata.back());

    // Test move construction/assignment
    {
        Byte*            orig_ptr = alloc.device_ref().data();
        DeviceAllocation other(std::move(alloc));
        EXPECT_EQ(128, other.size());
        EXPECT_EQ(0, alloc.size());
        EXPECT_EQ(orig_ptr, other.device_ref().data());
    }
}

TEST_F(DeviceAllocationTest, TEST_IF_CELER_DEVICE(empty))
{
    DeviceAllocation alloc{0};
    EXPECT_TRUE(alloc.empty());
    EXPECT_EQ(0, alloc.size());
    EXPECT_EQ(nullptr, alloc.device_ref().data());

    std::vector<Byte> newdata(alloc.size());
    alloc.copy_to_device(celeritas::make_span(newdata));
    alloc.copy_to_host(celeritas::make_span(newdata));
}
