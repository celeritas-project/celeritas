//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AsyncSnapshot.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/AsyncSnapshot.hh"

#include <atomic>
#include <thread>

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
/*!
 * Block work on a stream until released.
 *
 * A host callback enqueued on the stream spins until \c release is called,
 * so work enqueued after construction deterministically stays pending.
 * Destruction releases the stream and waits for the callback to return.
 */
class StreamBlocker
{
  public:
    explicit StreamBlocker(Stream& stream) : stream_{stream}
    {
        stream.launch_host_func(&StreamBlocker::spin, &released_);
    }

    ~StreamBlocker()
    {
        this->release();
        stream_.sync();
    }

    CELER_DELETE_COPY_MOVE(StreamBlocker);

    void release() { released_ = true; }

  private:
    Stream& stream_;
    std::atomic<bool> released_{false};

    static void spin(void* released)
    {
        while (!*static_cast<std::atomic<bool>*>(released))
        {
            std::this_thread::yield();
        }
    }
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
    EXPECT_EQ(0, snapshot.wait().first);
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

TEST(AsyncSnapshotTest, TEST_IF_CELER_DEVICE(device_pending))
{
    device().create_streams(1);
    StreamId const sid{0};

    Counts const init{5, 6};
    DeviceVector<Counts> src(1, sid);
    src.copy_to_device({&init, 1});

    AsyncSnapshot<Counts> snapshot;
    {
        StreamBlocker blocker{device().stream(sid)};
        snapshot.load_async(make_observer(src), sid);
        EXPECT_FALSE(snapshot.ready());
        if constexpr (CELERITAS_DEBUG)
        {
            // Reading or overwriting a pending load is checked
            EXPECT_THROW(snapshot.value(), DebugError);
            EXPECT_THROW(snapshot.assign(Counts{}), DebugError);
        }
        blocker.release();
        EXPECT_EQ(5, snapshot.wait().first);
        EXPECT_TRUE(snapshot.ready());
    }
    EXPECT_EQ(6, snapshot.value().second);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
