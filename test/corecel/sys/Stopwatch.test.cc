//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stopwatch.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/Stopwatch.hh"

#include <thread>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Sleep for the given duration.
 *
 * Sleeping only guarantees a *minimum* time delta. Note that std::chrono time
 * durations default to integer types.
 *
 * Example:
 * \code
   std::chrono::milliseconds actual_time
       = sleep_for(std::chrono::milliseconds(10));
   assert(actual_time >= 10);
 * \endcode
 * \return Elapsed time duration of actual sleep.
 */
template<class Rep, class Period>
inline auto sleep_for(std::chrono::duration<Rep, Period> const& duration)
    -> std::chrono::duration<Rep, Period>
{
    using Duration = std::chrono::duration<Rep, Period>;
    using Clock = std::chrono::high_resolution_clock;
    auto start = Clock::now();
    std::this_thread::sleep_for(duration);
    auto stop = Clock::now();
    return std::chrono::duration_cast<Duration>(stop - start);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(StopwatchTest, all)
{
    double tolerance = 0.1;

    // Start the clock, sleep, measure
    Stopwatch elapsed_time;
    auto actual_ms = sleep_for(std::chrono::milliseconds(50));
    double measured_s = elapsed_time();
    EXPECT_GE(measured_s, 0.05);
    EXPECT_SOFT_NEAR(actual_ms.count() * 0.001, measured_s, tolerance);
    EXPECT_LT(measured_s, 5.0);

    // Reset and immediately measure
    elapsed_time = {};
    measured_s = elapsed_time();
    EXPECT_LT(measured_s, 0.05);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
