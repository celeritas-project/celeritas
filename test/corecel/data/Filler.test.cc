//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Filler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Filler.hh"

#include <vector>

#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/sys/Device.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(FillerTest, host)
{
    std::vector<int> vec(128, 0);

    Filler<int, MemSpace::host> fill{987};
    fill(make_span(vec));
    for (int v : vec)
    {
        EXPECT_EQ(987, v);
    }
}

TEST(FillerTest, TEST_IF_CELER_DEVICE(device))
{
    celeritas::device().create_streams(3);

    DeviceVector<int> device_vec(128);
    std::size_t offset{0};
    std::size_t const subspan_size{32};
    for (auto s : {StreamId{}, StreamId{0}, StreamId{1}})
    {
        Filler<int, MemSpace::device> fill{987, s};
        fill(device_vec.device_ref().subspan(offset, subspan_size));
        offset += subspan_size;
    }

    {
        // GPU fill with default stream
        Filler<int, MemSpace::device> fill{987};
        fill(device_vec.device_ref().subspan(offset, subspan_size));
    }

    // Copy device --> host to check
    std::vector<int> test_vec(device_vec.size());
    {
        Copier<int, MemSpace::host> copy{make_span(test_vec)};
        copy(MemSpace::device, device_vec.device_ref());
    }
    for (int v : test_vec)
    {
        EXPECT_EQ(987, v);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
