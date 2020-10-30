//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Stopwatch.hh
//---------------------------------------------------------------------------//
#pragma once

#include <chrono>
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple timer.
 *
 * The stopwatch starts counting upward at construction and can be reset by
 * assigning a new stopwatch instance.
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
  public:
    // Start the count at construction
    inline Stopwatch();

    // Get the current elapsed time [s]
    inline real_type operator()() const;

  private:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    TimePoint start_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Stopwatch.i.hh"
