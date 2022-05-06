//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file base/Copier.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Copier.hh"

#include <vector>

#include "corecel/data/DeviceVector.hh"

#include "celeritas_test.hh"

using celeritas::Copier;
using celeritas::DeviceVector;
using celeritas::MemSpace;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CopierTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CopierTest, host)
{
    // Copy host --> host
    std::vector<int> src_vec(128, 1234);
    std::vector<int> dst_vec(src_vec.size() + 1);

    Copier<int, MemSpace::host> copy{celeritas::make_span(src_vec)};
    copy(MemSpace::host, {dst_vec.data() + 1, dst_vec.data() + dst_vec.size()});
    EXPECT_EQ(0, dst_vec.front());
    EXPECT_EQ(1234, dst_vec[1]);
    EXPECT_EQ(1234, dst_vec.back());
}

TEST_F(CopierTest, TEST_IF_CELER_DEVICE(device))
{
    // Copy host --> device
    std::vector<int> host_vec(128);
    host_vec.front() = 1;
    host_vec.back()  = 1234;
    DeviceVector<int> device_vec(host_vec.size());
    {
        Copier<int, MemSpace::host> copy{celeritas::make_span(host_vec)};
        copy(MemSpace::device, device_vec.device_ref());
    }

    // Copy device --> device
    DeviceVector<int> new_device_vec(host_vec.size());
    {
        Copier<int, MemSpace::device> copy{device_vec.device_ref()};
        copy(MemSpace::device, new_device_vec.device_ref());
    }

    // Copy device --> host
    std::vector<int> new_host_vec(host_vec.size());
    {
        Copier<int, MemSpace::device> copy{new_device_vec.device_ref()};
        copy(MemSpace::host, celeritas::make_span(new_host_vec));
    }
    EXPECT_EQ(1, new_host_vec.front());
    EXPECT_EQ(1234, new_host_vec.back());
}
