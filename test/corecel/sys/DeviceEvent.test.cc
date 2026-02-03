//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/DeviceEvent.hh"

#include <chrono>
#include <thread>

#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"

#include "celeritas_test.hh"

namespace chrono = std::chrono;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EventTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        auto& d = celeritas::device();
        if (d && d.num_streams() == 0)
        {
            d.create_streams(1);
        }
    }
};

//---------------------------------------------------------------------------//
// Helper function to be called on the device stream
void my_host_kernel_impl(int duration_ms)
{
    std::this_thread::sleep_for(chrono::milliseconds(duration_ms));
}

void my_host_kernel(void* user_data)
{
    auto* duration_ms = static_cast<int*>(user_data);
    return my_host_kernel_impl(*duration_ms);
}

Stream& get_stream(StreamId sid)
{
    if (auto& d = celeritas::device())
    {
        CELER_EXPECT(sid < d.num_streams());
        return d.stream(sid);
    }
    else
    {
        // Return null stream
        static Stream s;
        return s;
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EventTest, construct_from_stream_id)
{
    StreamId stream_id{0};
    DeviceEvent event(stream_id);

    // Event should be ready immediately after construction
    EXPECT_TRUE(event.ready());
}

TEST_F(EventTest, record_and_query)
{
    Stream& s = get_stream(StreamId{0});
    DeviceEvent event(s);

    // Note that the lifetime of the argument must be longer than the
    // stack, since the function is called asynchronously on another thread
    static int const delay_ms = 100;

    // Launch a delayed host function on the stream
    s.launch_host_func(my_host_kernel, const_cast<int*>(&delay_ms));

    // Record the event after the delayed function
    event.record();

    // Event should not be ready if running asynchronously, but will be ready
    // if on host
    EXPECT_EQ(!CELER_USE_DEVICE, event.ready());

    // Sync should block until the delay is complete
    auto start = chrono::steady_clock::now();
    event.sync();
    auto duration = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - start);

    // Should have waited at least the delay time
    if (CELER_USE_DEVICE)
    {
        EXPECT_GE(duration.count(), delay_ms);
    }
    EXPECT_TRUE(event.ready());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
