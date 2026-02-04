//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/DeviceEvent.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/DeviceEvent.hh"

#include <chrono>
#include <thread>

#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/Stream.hh"

#include "celeritas_test.hh"

namespace chrono = std::chrono;
constexpr double ms_to_s = 0.001;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class DeviceEventTest : public ::celeritas::test::Test
{
};

//---------------------------------------------------------------------------//
// Helper function to be called on the device stream
void my_host_kernel_impl(int duration_ms)
{
    std::this_thread::sleep_for(chrono::milliseconds(duration_ms));
}

void my_host_kernel(void* user_data)
{
    CELER_EXPECT(user_data);
    auto* duration_ms = static_cast<int*>(user_data);
    return my_host_kernel_impl(*duration_ms);
}

int g_value{0};

void set_value(void* user_data)
{
    CELER_EXPECT(user_data);
    g_value = *static_cast<int*>(user_data);
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

    // Test implicit construction
    stream = nullptr;
    event = nullptr;
}

TEST_F(DeviceEventTest, TEST_IF_CELER_DEVICE(single_stream))
{
    Stream s(device());
    ASSERT_TRUE(s);
    DeviceEvent e(device());
    ASSERT_TRUE(e);

    for (int i = 0; i < 2; ++i)
    {
        // Note that the lifetime of user kernel arguments must be longer than
        // the stack, since the function is called asynchronously on another
        // thread
        static int const delay_ms = 50;
        constexpr double ms_to_s = 0.001;

        // Launch a delayed host function on the stream
        Stopwatch get_time;
        s.launch_host_func(my_host_kernel, const_cast<int*>(&delay_ms));
        // Record the event after the delayed function
        e.record(s);

        // Event should not be ready if running asynchronously
        EXPECT_FALSE(e.ready());

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

    // Test that moving works
    Stream s2(std::move(s));
    EXPECT_TRUE(s2);
    EXPECT_FALSE(s);

    DeviceEvent e2(std::move(e));
    EXPECT_TRUE(e2);
    EXPECT_FALSE(e);
}

TEST_F(DeviceEventTest, TEST_IF_CELER_DEVICE(multi_stream))
{
    Stream s1(device());
    ASSERT_TRUE(s1);
    DeviceEvent e1(device());
    ASSERT_TRUE(e1);

    static int const delay_ms = 150;

    // Launch a delayed host function on the stream
    Stopwatch get_time;
    s1.launch_host_func(my_host_kernel, const_cast<int*>(&delay_ms));

    // Check that it's lagged and reset the timer to reduce jitter from ROCm
    // startup costs
    EXPECT_LT(get_time(), delay_ms * ms_to_s);
    get_time = {};
    e1.record(s1);

    Stream s2(device());
    EXPECT_LT(get_time(), delay_ms * ms_to_s);
    // Create an event for stream 2
    DeviceEvent e2{device()};
    EXPECT_LT(get_time(), delay_ms * ms_to_s);

    // Tell stream2 to wait until stream 1's kernel is done (i.e., the
    // stream record stored in 'e')
    s2.wait(e1);

    // Then after waiting, launch a kernel on stream 2
    g_value = 0;
    static int const new_g_value{3};
    s2.launch_host_func(set_value, const_cast<int*>(&new_g_value));
    e2.record(s2);

    // Execution should be delayed
    EXPECT_EQ(0, g_value);
    EXPECT_FALSE(e1.ready());
    EXPECT_FALSE(e2.ready());
    EXPECT_LT(get_time(), delay_ms * ms_to_s);

    // Wait until first stream finished its kernel
    e1.sync();
    // Now g_value has *possibly* been updated, but we can't know for
    // sure due to multithreading delaying further
    if (e2.ready())
    {
        CELER_LOG(debug) << "execution completed already";
    }
    // Wait until the second event is done, i.e., g_value is updated
    e2.sync();
    EXPECT_EQ(new_g_value, g_value);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
