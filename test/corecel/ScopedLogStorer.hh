//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/ScopedLogStorer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/io/LoggerTypes.hh"

namespace celeritas
{
class Logger;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Log handle for saving messages for testing.
 *
 * Temporarily replace the given logger with this function. This removes ANSI
 * sequences and replaces pointer-like strings with 0x0.
 *
 * \code
    ScopedLogStorer store_log_;
    world_logger() = Logger(store_log_);
   \endcode
 */
class ScopedLogStorer
{
  public:
    //!@{
    //! \name Type aliases
    using VecString = std::vector<std::string>;
    //!@}

  public:
    // Construct reference to log to temporarily replace
    ScopedLogStorer(Logger* orig, LogLevel min_level);

    // Construct reference with default level
    explicit ScopedLogStorer(Logger* orig);

    // Restore original logger on destruction
    ~ScopedLogStorer();

    // Save a log message
    void operator()(Provenance, LogLevel lev, std::string msg);

    // Get saved messages
    VecString const& messages() const { return messages_; }

    // Get corresponding log levels
    VecString const& levels() const { return levels_; }

    // Print expected results to stdout
    void print_expected() const;

  private:
    Logger* logger_;
    std::unique_ptr<Logger> saved_logger_;
    VecString messages_;
    VecString levels_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
