//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LogHandlers.hh
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
 * Celeritas-style multithreaded log handle that writes to std::clog.
 */
class SelfLogHandler
{
  public:
    // Construct from number of threads and global comm
    explicit SelfLogHandler(int num_threads);

    // Write a log message
    void operator()(LogProvenance prov, LogLevel lev, std::string msg);

  private:
    int rank_{};
    int size_{};
    int num_threads_{};
};

//---------------------------------------------------------------------------//

// Create a handler for "everyone logs the same" messages
LogHandler make_world_handler();

// Create a handler for thread-local messages
LogHandler make_self_handler(int num_threads);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
