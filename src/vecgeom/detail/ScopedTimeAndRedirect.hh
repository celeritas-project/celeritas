//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedTimeAndRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "base/ScopedStreamRedirect.hh"
#include "base/Stopwatch.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * At end of scope, print elapsed time and captured cout/cerr.
 *
 * \code
    {
        ScopedTimeAndRedirect temp_;
        vecgeom::DoNoisyAndLongStuff();
    }
   \endcode
 *
 * During scope, you should be sure *NOT* to call the logger, which by default
 * prints to cerr.
 */
class ScopedTimeAndRedirect
{
  public:
    ScopedTimeAndRedirect();
    ~ScopedTimeAndRedirect();

  private:
    std::unique_ptr<ScopedStreamRedirect> stdout_;
    std::unique_ptr<ScopedStreamRedirect> stderr_;
    Stopwatch                             get_time_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
