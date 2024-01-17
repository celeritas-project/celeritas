//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedTimeLog.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
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

    // Construct with a reference to a particular logger (e.g. thread-local)
    explicit inline ScopedTimeLog(Logger* dest);

    // Construct with manual threshold for printing time
    explicit inline ScopedTimeLog(double min_print_sec);

    // Construct with logger and time threshold
    inline ScopedTimeLog(Logger* dest, double min_print_sec);

    // Print on destruction
    inline ~ScopedTimeLog();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedTimeLog);
    //!@}

  private:
    Logger* logger_{nullptr};
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
    CELER_EXPECT(min_print_sec >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to a particular logger.
 */
ScopedTimeLog::ScopedTimeLog(Logger* dest) : logger_(dest)
{
    CELER_EXPECT(logger_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to a particular logger.
 */
ScopedTimeLog::ScopedTimeLog(Logger* dest, double min_print_sec)
    : logger_(dest), min_print_sec_(min_print_sec)
{
    CELER_EXPECT(logger_);
    CELER_EXPECT(min_print_sec >= 0);
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
        auto msg = [this] {
            if (!logger_)
            {
                return CELER_LOG(diagnostic);
            }
            return (*logger_)(CELER_CODE_PROVENANCE, LogLevel::diagnostic);
        }();
        msg << color_code('x') << "... " << time_sec << " s" << color_code(' ');
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
