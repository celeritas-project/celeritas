//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "corecel/io/StringUtils.hh"

#include "Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Log messages separated by lines
void log_messages(std::string const& label,
                  LogLevel level,
                  std::stringstream& ss)
{
    Logger& celer_log = celeritas::world_logger();
    std::string templine;
    while (std::getline(ss, templine, '\n'))
    {
        while (!templine.empty() && is_ignored_trailing(templine.back()))
        {
            templine.pop_back();
        }
        if (!templine.empty())
        {
            celer_log({label, 0}, level) << templine;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace

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
    // Capture output streams and reset *before* logging
    std::stringstream sout = std::move(stdout_->get());
    std::stringstream serr = std::move(stderr_->get());
    stdout_.reset();
    stderr_.reset();

    log_messages(label_, LogLevel::diagnostic, sout);
    log_messages(label_, LogLevel::warning, serr);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
