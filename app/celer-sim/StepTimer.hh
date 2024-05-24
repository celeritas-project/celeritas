//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/StepTimer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/sys/Stopwatch.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Optionally append a time at every call.
 */
class StepTimer
{
  public:
    //!@{
    //! \name Type aliases
    using VecDbl = std::vector<double>;
    //!@}

  public:
    // Construct with a pointer to the times being appended, possibly null
    explicit inline StepTimer(VecDbl* times);

    // If enabled, push back the time and reset the timer
    inline void operator()();

  private:
    VecDbl* times_;
    Stopwatch get_step_time_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a pointer to the times being appended.
 *
 * If this is null, the stopwatch will not be enabled.
 */
StepTimer::StepTimer(VecDbl* times) : times_{times} {}

//---------------------------------------------------------------------------//
/*!
 * Push back the time and reset the timer, if requested.
 */
void StepTimer::operator()()
{
    if (times_)
    {
        times_->push_back(get_step_time_());
        get_step_time_ = {};
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
