//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Copier.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Copier.hh"

#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(CopierTest, host)
{
    // Copy host --> host
    std::vector<int> src_vec(128, 1234);
    std::vector<int> dst_vec(src_vec.size() + 1);

    Copier copy{ObserverPtr<int, MemSpace::host>{dst_vec.data() + 1},
                src_vec.size(),
                StreamId{}};
    copy(MemSpace::host, make_span(src_vec));
    EXPECT_EQ(0, dst_vec.front());
    EXPECT_EQ(1234, dst_vec[1]);
    EXPECT_EQ(1234, dst_vec.back());
}

TEST(CopierTest, TEST_IF_CELER_DEVICE(device))
{
    // Copy host --> device
    std::vector<int> host_vec(128);
    host_vec.front() = 1;
    host_vec.back() = 1234;
    DeviceVector<int> device_vec(host_vec.size());
    {
        Copier<int, MemSpace::device> copy{device_vec.device_ref(), StreamId{}};
        copy(MemSpace::host, make_span(host_vec));
    }

    // Copy device --> device
    DeviceVector<int> new_device_vec(host_vec.size());
    {
        Copier<int, MemSpace::device> copy{new_device_vec.device_ref(),
                                           StreamId{}};
        copy(MemSpace::device, device_vec.device_ref());
    }

    // Copy device --> host
    std::vector<int> new_host_vec(host_vec.size());
    {
        Copier<int, MemSpace::host> copy{make_span(new_host_vec), StreamId{}};
        copy(MemSpace::device, new_device_vec.device_ref());
    }
    EXPECT_EQ(1, new_host_vec.front());
    EXPECT_EQ(1234, new_host_vec.back());
}

TEST(CopyBytesTest, TEST_IF_CELER_DEVICE(device_memcpy_stream))
{
    device().create_streams(1);
    Stream& stream = device().memcpy_stream(StreamId{0});

    std::vector<int> host_vec{0, 1, 2, 3, 4};
    DeviceVector<int> device_vec(host_vec.size());
    copy_bytes<MemSpace::device, MemSpace::host>(make_observer(device_vec),
                                                 make_observer(host_vec.data()),
                                                 host_vec.size() * sizeof(int),
                                                 stream);

    // Pinned destination so the device-to-host copy is truly asynchronous
    std::vector<int, PinnedAllocator<int>> result(host_vec.size());
    copy_bytes<MemSpace::host, MemSpace::device>(make_observer(result.data()),
                                                 make_observer(device_vec),
                                                 result.size() * sizeof(int),
                                                 stream);
    stream.sync();
    EXPECT_VEC_EQ(host_vec, std::vector<int>(result.begin(), result.end()));
}

TEST(LoadSyncTest, TEST_IF_CELER_DEVICE(device))
{
    // Initialize data on device
    std::vector<int> host_vec{0, 1, 2, 3, 4};
    DeviceVector<int> device_vec(host_vec.size());
    device_vec.copy_to_device(make_span(host_vec));

    // Copy elements to host on a stream
    device().create_streams(1);
    StreamId stream{0};
    for (int i : range(host_vec.size()))
    {
        ObserverPtr<int, MemSpace::device> p{device_vec.data() + i};
        EXPECT_EQ(i, load_sync(p, stream));
    }
    {
        // Test const overload
        ObserverPtr<int const, MemSpace::device> p{device_vec.data() + 2};
        EXPECT_EQ(2, load_sync(p, stream));

        // Test without stream
        EXPECT_EQ(2, load_sync(p));
    }
}

TEST(LoadSyncTest, host)
{
    int value = 1234;
    EXPECT_EQ(1234, load_sync(ObserverPtr<int, MemSpace::host>{&value}, {}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
