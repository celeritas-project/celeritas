//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceAllocation.test.cc
//---------------------------------------------------------------------------//
#include "base/DeviceAllocation.hh"

#include "celeritas_test.hh"
#include "base/Span.hh"

using celeritas::Byte;
using celeritas::DeviceAllocation;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DeviceAllocationTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DeviceAllocationTest, all)
{
    DeviceAllocation alloc;
    EXPECT_EQ(0, alloc.size());
    EXPECT_TRUE(alloc.empty());

#if !CELERITAS_USE_CUDA
    // Can't allocate
    EXPECT_THROW(DeviceAllocation(1234), celeritas::DebugError);
    cout << "CUDA is disabled; skipping remainder of test." << endl;
    return;
#endif

    alloc = DeviceAllocation(1024);
    EXPECT_EQ(1024, alloc.size());
    EXPECT_FALSE(alloc.empty());

    {
        DeviceAllocation other(128);
        Byte*            orig_alloc = alloc.device_pointers().data();
        Byte*            orig_other = other.device_pointers().data();
        swap(alloc, other);
        EXPECT_EQ(1024, other.size());
        EXPECT_EQ(orig_other, alloc.device_pointers().data());
        EXPECT_EQ(orig_alloc, other.device_pointers().data());
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
        Byte*            orig_ptr = alloc.device_pointers().data();
        DeviceAllocation other(std::move(alloc));
        EXPECT_EQ(128, other.size());
        EXPECT_EQ(0, alloc.size());
        EXPECT_EQ(orig_ptr, other.device_pointers().data());
    }
}
