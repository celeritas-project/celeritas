//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedTimeLog.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/Stopwatch.hh"

#include "ColorUtils.hh"
#include "Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Print the elapsed time since construction when destructed.
 *
 * An optional construction argument specifies the minimum time needed to
 * bother printing.
 * \code
     {
         CELER_LOG(info) << "Doing something expensive";
         ScopedTimeLog scoped_time;
         do_something_expensive();
     }
   \endcode
 */
class ScopedTimeLog
{
  public:
    // Construct with default threshold of 0.01 seconds
    inline ScopedTimeLog() = default;

    // Construct with manual threshold for printing time
    explicit inline ScopedTimeLog(double min_print_sec);

    // Print on destruction
    inline ~ScopedTimeLog();

  private:
    double min_print_sec_{0.01};
    Stopwatch get_time_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with a manual threshold for printing time.
 */
ScopedTimeLog::ScopedTimeLog(double min_print_sec)
    : min_print_sec_(min_print_sec)
{
    CELER_ASSERT(min_print_sec >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Print large enough times when exiting scope.
 */
ScopedTimeLog::~ScopedTimeLog()
{
    double time_sec = get_time_();
    if (time_sec > min_print_sec_)
    {
        using celeritas::color_code;
        CELER_LOG(diagnostic) << color_code('x') << "... " << time_sec << " s"
                              << color_code(' ');
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
