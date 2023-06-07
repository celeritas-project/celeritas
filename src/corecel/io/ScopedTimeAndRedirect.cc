//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedTimeAndRedirect.cc
//---------------------------------------------------------------------------//
#include "ScopedTimeAndRedirect.hh"

#include <iostream>
#include <utility>

#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/io/ScopedStreamRedirect.hh"

#include "Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Redirect cout/cerr on construction, and start timer implicitly.
 */
ScopedTimeAndRedirect::ScopedTimeAndRedirect(std::string label)
    : stdout_{std::make_unique<ScopedStreamRedirect>(&std::cout)}
    , stderr_{std::make_unique<ScopedStreamRedirect>(&std::cerr)}
    , label_{std::move(label)}
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
        celer_log({label_, 0}, LogLevel::diagnostic) << sout;
    }

    if (!serr.empty())
    {
        celer_log({label_, 0}, LogLevel::warning) << serr;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
