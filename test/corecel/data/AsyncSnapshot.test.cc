//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncSnapshot.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/AsyncSnapshot.hh"

#include "corecel/data/DeviceVector.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceEvent.hh"
#include "corecel/sys/Stream.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct Counts
{
    int first{0};
    int second{0};
};

//---------------------------------------------------------------------------//

TEST(AsyncSnapshotTest, host)
{
    AsyncSnapshot<Counts> snapshot{Counts{1, 2}};
    EXPECT_EQ(1, snapshot.value().first);
    EXPECT_EQ(2, snapshot.value().second);

    Counts src{3, 4};
    snapshot.copy_async(make_observer(&src), StreamId{});
    src.first = 10;
    EXPECT_EQ(3, snapshot.value().first);
    EXPECT_EQ(4, snapshot.value().second);

    // Write through the result reference
    Counts const other{7, 8};
    copy_to_host_async(make_observer(&other), snapshot.ref(), StreamId{});
    EXPECT_EQ(7, snapshot.value().first);

    snapshot.assign(Counts{});
    EXPECT_EQ(0, snapshot.value().first);
}

TEST(AsyncSnapshotTest, TEST_IF_CELER_DEVICE(device))
{
    device().create_streams(1);
    StreamId const sid{0};

    Counts const init{5, 6};
    DeviceVector<Counts> src(1, sid);
    src.copy_to_device({&init, 1});

    AsyncSnapshot<Counts> snapshot;
    snapshot.copy_async(make_observer(src), sid);
    DeviceEvent done{device()};
    done.record(device().stream(sid));
    done.sync();
    EXPECT_EQ(5, snapshot.value().first);
    EXPECT_EQ(6, snapshot.value().second);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
