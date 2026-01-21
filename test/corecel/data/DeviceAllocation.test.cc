//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/DeviceAllocation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/DeviceAllocation.hh"

#include <algorithm>

#include "corecel/cont/Span.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

// NOTE: don't have 'device' in the name here
TEST(ConstructionTest, should_work_always)
{
    DeviceAllocation alloc;
    EXPECT_EQ(0, alloc.size());
    EXPECT_EQ(nullptr, alloc.data());
    EXPECT_TRUE(alloc.empty());
    EXPECT_EQ(StreamId{}, alloc.stream_id());

    DeviceAllocation alloc2;
    swap(alloc, alloc2);
    EXPECT_TRUE(alloc.empty());
    EXPECT_TRUE(alloc.device_ref().empty());
}

#if !CELER_USE_DEVICE
TEST(ConstructionTest, nodevice)
#else
TEST(ConstructionTest, DISABLED_nodevice)
#endif
{
    // Can't allocate
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(DeviceAllocation(1234), DebugError);
    }
    else
    {
        EXPECT_THROW(DeviceAllocation(1234), RuntimeError);
    }
}

TEST(DeviceAllocationTest, TEST_IF_CELER_DEVICE(device))
{
    DeviceAllocation alloc(1024);
    EXPECT_EQ(1024, alloc.size());
    EXPECT_FALSE(alloc.empty());

    {
        DeviceAllocation other(128);
        std::byte* orig_alloc = alloc.device_ref().data();
        std::byte* orig_other = other.device_ref().data();
        swap(alloc, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, alloc.device_ref().data());
        EXPECT_EQ(orig_alloc, other.device_ref().data());
    }
    std::vector<std::byte> data(alloc.size());
    ASSERT_EQ(128, data.size());

    data.front() = std::byte(1);
    data.back() = std::byte(127);

    alloc.copy_to_device(make_span(data));

    std::vector<std::byte> newdata(alloc.size());
    ASSERT_EQ(128, newdata.size());
    alloc.copy_to_host(make_span(newdata));
    EXPECT_EQ(std::byte(1), newdata.front());
    EXPECT_EQ(std::byte(127), newdata.back());

    // Test move construction/assignment
    {
        std::byte* orig_ptr = alloc.device_ref().data();
        DeviceAllocation other(std::move(alloc));
        EXPECT_EQ(128, other.size());
        EXPECT_EQ(0, alloc.size());
        EXPECT_EQ(orig_ptr, other.device_ref().data());
    }
}

TEST(DeviceAllocationTest, TEST_IF_CELER_DEVICE(empty))
{
    DeviceAllocation alloc{0};
    EXPECT_TRUE(alloc.empty());
    EXPECT_EQ(0, alloc.size());
    EXPECT_EQ(nullptr, alloc.device_ref().data());

    std::vector<std::byte> newdata(alloc.size());
    alloc.copy_to_device(make_span(newdata));
    alloc.copy_to_host(make_span(newdata));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
