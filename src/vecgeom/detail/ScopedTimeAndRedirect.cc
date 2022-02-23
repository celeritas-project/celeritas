//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedTimeAndRedirect.cc
//---------------------------------------------------------------------------//
#include "ScopedTimeAndRedirect.hh"

#include <iostream>

#include "base/ColorUtils.hh"
#include "comm/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Redirect cout/cerr on construction, and start timer implicitly.
 */
ScopedTimeAndRedirect::ScopedTimeAndRedirect()
    : stdout_{std::make_unique<ScopedStreamRedirect>(&std::cout)}
    , stderr_{std::make_unique<ScopedStreamRedirect>(&std::cerr)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Print any stored output/error messages on destruct.
 *
 * Note that these will only print on MPI rank 0, not on every process.
 */
ScopedTimeAndRedirect::~ScopedTimeAndRedirect()
{
    std::string sout = stdout_->str();
    stdout_.reset();
    std::string serr = stderr_->str();
    stderr_.reset();

    Logger& celer_log = celeritas::world_logger();
    if (!sout.empty())
    {
        celer_log({"vecgeom", 0}, LogLevel::diagnostic) << sout;
    }

    if (!serr.empty())
    {
        celer_log({"vecgeom", 0}, LogLevel::warning) << serr;
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
