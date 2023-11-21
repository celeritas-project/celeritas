//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LocalLogger.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Assert.hh"
#include "corecel/io/LoggerTypes.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Celeritas-style multithreaded logger that writes to std::clog.
 */
class LocalLogger
{
  public:
    // Construct with defaults
    explicit LocalLogger(int num_threads) : num_threads_(num_threads)
    {
        CELER_EXPECT(num_threads_ > 0);
    }

    // Write a log message
    void operator()(Provenance prov, LogLevel lev, std::string msg);

  private:
    int num_threads_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
