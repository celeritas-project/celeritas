//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stopwatch.hh
//---------------------------------------------------------------------------//
#pragma once

#include <chrono>

#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple timer.
 *
 * The stopwatch starts counting upward at construction and can be reset by
 * assigning a new stopwatch instance. It needs to be enabled by setting the
 * CELER_ENABLE_STOPWATCH environment variable, otherwise returns no elapsed
 * time.
 *
 * \code
    Stopwatch get_elapsed_time;
    // ...
    double time = get_elapsed_time();
    // Reset the stopwatch
    get_elapsed_time = {};
   \endcode
 */
class Stopwatch
{
  private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

  public:
    // Start the count at construction
    inline Stopwatch();

    // Get the current elapsed time [s]
    inline double operator()() const;
    inline static TimePoint now();

  private:
    TimePoint start_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Start the count at construction.
 */
Stopwatch::Stopwatch() : start_(Stopwatch::now()) {}

//---------------------------------------------------------------------------//
/*!
 * Get the current elapsed time in seconds.
 */
double Stopwatch::operator()() const
{
    using DurationSec = std::chrono::duration<double>;

    auto duration = Stopwatch::now() - start_;
    return std::chrono::duration_cast<DurationSec>(duration).count();
}

inline auto Stopwatch::now() -> TimePoint
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_STOPWATCH").empty())
        {
            CELER_LOG(info) << "Enabling timing information since the "
                               "'CELER_ENABLE_STOPWATCH' "
                               "environment variable is present and non-empty";
            return true;
        }
        return false;
    }();
    return result ? Clock::now() : TimePoint::min();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
