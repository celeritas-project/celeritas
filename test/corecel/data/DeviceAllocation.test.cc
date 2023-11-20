//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    EXPECT_TRUE(alloc.empty());
}

TEST(ConstructionTest, nocuda)
{
#if !CELER_USE_DEVICE
    // Can't allocate
    EXPECT_THROW(DeviceAllocation(1234), DebugError);
#else
    GTEST_SKIP() << "CUDA is enabled";
#endif
}

TEST(DeviceAllocationTest, TEST_IF_CELER_DEVICE(device))
{
    DeviceAllocation alloc(1024);
    EXPECT_EQ(1024, alloc.size());
    EXPECT_FALSE(alloc.empty());

    {
        DeviceAllocation other(128);
        Byte* orig_alloc = alloc.device_ref().data();
        Byte* orig_other = other.device_ref().data();
        swap(alloc, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, alloc.device_ref().data());
        EXPECT_EQ(orig_alloc, other.device_ref().data());
    }
    EXPECT_EQ(128, alloc.size());

    std::vector<Byte> data(alloc.size());
    data.front() = Byte(1);
    data.back() = Byte(127);

    alloc.copy_to_device(make_span(data));

    std::vector<Byte> newdata(alloc.size());
    alloc.copy_to_host(make_span(newdata));
    EXPECT_EQ(Byte(1), newdata.front());
    EXPECT_EQ(Byte(127), newdata.back());

    // Test move construction/assignment
    {
        Byte* orig_ptr = alloc.device_ref().data();
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

    std::vector<Byte> newdata(alloc.size());
    alloc.copy_to_device(make_span(newdata));
    alloc.copy_to_host(make_span(newdata));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
