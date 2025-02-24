//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    //! Construct from number of threads
    explicit LocalLogger(unsigned int num_threads) : num_threads_(num_threads)
    {
        CELER_EXPECT(num_threads_ > 0);
    }

    // Write a log message
    void operator()(LogProvenance prov, LogLevel lev, std::string msg);

  private:
    unsigned int num_threads_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
