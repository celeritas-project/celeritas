//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncSnapshot.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/AsyncSnapshot.hh"

#include "corecel/data/DeviceVector.hh"
#include "corecel/sys/Device.hh"
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
    EXPECT_TRUE(snapshot.ready());
    EXPECT_EQ(1, snapshot.value().first);
    EXPECT_EQ(2, snapshot.value().second);

    // Host loads complete immediately
    Counts src{3, 4};
    snapshot.load_async(make_observer(&src), StreamId{});
    EXPECT_TRUE(snapshot.ready());
    src.first = 10;
    EXPECT_EQ(3, snapshot.value().first);
    EXPECT_EQ(4, snapshot.value().second);

    snapshot.sync();
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
    snapshot.load_async(make_observer(src), sid);
    snapshot.sync();
    EXPECT_TRUE(snapshot.ready());
    EXPECT_EQ(5, snapshot.value().first);
    EXPECT_EQ(6, snapshot.value().second);

    // Reloading re-records the same event
    Counts const update{7, 8};
    src.copy_to_device({&update, 1});
    snapshot.load_async(make_observer(src), sid);
    snapshot.sync();
    EXPECT_EQ(7, snapshot.value().first);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
