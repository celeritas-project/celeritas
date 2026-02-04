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

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class DeviceEventTest : public ::celeritas::test::Test
{
  public:
    void run(Stream& s, DeviceEvent& de) const;
    void run_multistream(Stream& s1, Stream& s2, DeviceEvent& e) const;
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
//! Test event synchronization using host kernels
void DeviceEventTest::run(Stream& s, DeviceEvent& e) const
{
    CELER_EXPECT(static_cast<bool>(s) == static_cast<bool>(e));

    // Note that the lifetime of user kernel arguments must be longer than the
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
void DeviceEventTest::run_multistream(Stream& s1,
                                      Stream& s2,
                                      DeviceEvent& e) const
{
    CELER_EXPECT(static_cast<bool>(s1) == static_cast<bool>(e));
    CELER_EXPECT(static_cast<bool>(s2) == static_cast<bool>(s1));

    static int const delay_ms = 100;
    constexpr double ms_to_s = 0.001;

    // Launch a delayed host function on the stream
    Stopwatch get_time;
    s1.launch_host_func(my_host_kernel, const_cast<int*>(&delay_ms));
    if (!e)
    {
        // No device: function executes instantaneously
        EXPECT_GE(get_time(), delay_ms * ms_to_s);
    }
    e.record();

    // Create an event for stream 2
    DeviceEvent e2{s2};
    if (e)
    {
        EXPECT_LT(get_time(), delay_ms * ms_to_s);
    }
    // Tell stream2 to wait until stream 1's kernel is done (i.e., the stream
    // record stored in 'e')
    stream_wait_event(s2, e);

    // Then after waiting, launch a kernel on stream 2
    g_value = 0;
    static int const new_g_value{3};
    s2.launch_host_func(set_value, const_cast<int*>(&new_g_value));
    e2.record();
    if (e)
    {
        // Execution should be delayed
        EXPECT_EQ(0, g_value);
        EXPECT_FALSE(e.ready());
        EXPECT_FALSE(e2.ready());
        EXPECT_LT(get_time(), delay_ms * ms_to_s);
    }
    else
    {
        // Stream operations execute immediately when device disabled
        EXPECT_EQ(new_g_value, g_value);
        EXPECT_TRUE(e.ready());
        EXPECT_TRUE(e2.ready());
    }

    // Wait until first stream finished its kernel
    e.sync();
    // Now g_value has *possibly* been updated, but we can't know for sure due
    // to multithreading delaying further
    if (!e2 && e2.ready())
    {
        CELER_LOG(debug) << "execution completed already";
    }
    // Wait until the second event is done, i.e., g_value is updated
    e2.sync();
    EXPECT_EQ(new_g_value, g_value);
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
    Stream s2{nullptr};
    this->run_multistream(stream, s2, event);

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

    // Run with multiple streams
    stream = Stream{celeritas::device()};
    this->run_multistream(s2, stream, e2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
