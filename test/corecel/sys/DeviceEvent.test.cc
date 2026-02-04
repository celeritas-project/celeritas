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
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/Stream.hh"

#include "celeritas_test.hh"

namespace chrono = std::chrono;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class DeviceEventTest : public ::celeritas::test::Test
{
  public:
    void run(Stream& s, DeviceEvent& de) const;
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

//---------------------------------------------------------------------------//
void DeviceEventTest::run(Stream& s, DeviceEvent& e) const
{
    // Note that the lifetime of the argument must be longer than the
    // stack, since the function is called asynchronously on another thread
    static int const delay_ms = 50;
    constexpr double ms_to_s = 0.001;

    // Launch a delayed host function on the stream
    Stopwatch get_time;
    s.launch_host_func(my_host_kernel, const_cast<int*>(&delay_ms));
    if (!e)
    {
        // No device: function executes instantaneously
        EXPECT_GE(get_time(), delay_ms * ms_to_s);
    }

    // Record the event after the delayed function
    e.record();

    if (e)
    {
        // Event should not be ready if running asynchronously
        EXPECT_FALSE(e.ready());
    }
    else
    {
        // Event executed immediately
        EXPECT_TRUE(e.ready());
    }

    // Sync should block until the delay is complete
    get_time = {};
    e.sync();

    if (e)
    {
        // Should have waited at least the delay time
        EXPECT_GE(get_time(), delay_ms * ms_to_s);
    }
    EXPECT_TRUE(e.ready());
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(DeviceEventTest, host)
{
    Stream stream{nullptr};
    EXPECT_FALSE(stream);

    DeviceEvent event{nullptr};
    EXPECT_FALSE(event);

    // Event should be ready immediately after construction
    EXPECT_TRUE(event.ready());

    this->run(stream, event);

    // Test implicit construction
    stream = nullptr;
    event = nullptr;
}

TEST_F(DeviceEventTest, TEST_IF_CELER_DEVICE(device))
{
    Stream stream(celeritas::device());
    ASSERT_TRUE(stream);
    DeviceEvent event(stream);
    ASSERT_TRUE(event);

    // Run an event
    this->run(stream, event);

    // Reuse the event
    this->run(stream, event);

    // Test that moving works
    Stream s2(std::move(stream));
    EXPECT_TRUE(s2);
    EXPECT_FALSE(stream);

    DeviceEvent e2(std::move(event));
    EXPECT_TRUE(e2);
    EXPECT_FALSE(event);

    // Run with the new event
    this->run(s2, e2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
