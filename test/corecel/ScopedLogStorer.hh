//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/ScopedLogStorer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <ostream>
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
    ScopedLogStorer scoped_log_{&celeritas::world_logger()};
    CELER_LOG(info) << "captured";
    scoped_log_.print_expected();
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
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

    //!@{
    //! Disallow move/copy
    ScopedLogStorer(ScopedLogStorer const&) = delete;
    ScopedLogStorer(ScopedLogStorer&&) = delete;
    ScopedLogStorer& operator=(ScopedLogStorer const&) = delete;
    ScopedLogStorer& operator=(ScopedLogStorer&&) = delete;
    //!@}

    // Restore original logger on destruction
    ~ScopedLogStorer();

    // Save a log message
    void operator()(Provenance, LogLevel lev, std::string msg);

    //! Whether no messages were stored
    bool empty() const { return messages_.empty(); }

    //! Get saved messages
    VecString const& messages() const { return messages_; }

    //! Get corresponding log levels
    VecString const& levels() const { return levels_; }

    // Print expected results to stdout
    void print_expected() const;

    //! Clear results
    void clear()
    {
        messages_.clear();
        levels_.clear();
    }

  private:
    Logger* logger_;
    std::unique_ptr<Logger> saved_logger_;
    VecString messages_;
    VecString levels_;
};

// Print expected results
std::ostream& operator<<(std::ostream& os, ScopedLogStorer const& logs);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
