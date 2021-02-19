//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
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
{
    stdout_ = std::make_unique<ScopedStreamRedirect>(&std::cout);
    stderr_ = std::make_unique<ScopedStreamRedirect>(&std::cerr);
}

//---------------------------------------------------------------------------//
/*!
 * Print any stored output/error messages on destruct.
 *
 * Note that these will only print on MPI rank 0, not on every process.
 */
ScopedTimeAndRedirect::~ScopedTimeAndRedirect()
{
    // Save timer and clear redirects before printing
    auto time_sec = get_time_();

    std::string sout = stdout_->str();
    stdout_.reset();
    std::string serr = stderr_->str();
    stderr_.reset();

    if (!sout.empty())
    {
        ::celeritas::world_logger()({"vecgeom", 0}, LogLevel::diagnostic)
            << sout;
    }

    if (!serr.empty())
    {
        ::celeritas::world_logger()({"vecgeom", 0}, LogLevel::warning) << serr;
    }

    if (time_sec > 0.01)
    {
        using celeritas::color_code;
        CELER_LOG(diagnostic) << color_code('x') << "... " << time_sec << " s"
                              << color_code(' ');
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
