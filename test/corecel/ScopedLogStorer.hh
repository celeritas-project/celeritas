//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/ScopedLogStorer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
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
 * You can use the \c CELER_LOG_SCOPED environment variable to print the
 * captured log messages as they are written.
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

    // Construct null storer for disassociating before destruction
    ScopedLogStorer();

    // Restore original logger on destruction
    ~ScopedLogStorer();

    //! Prevent copying but allow moving
    CELER_DEFAULT_MOVE_DELETE_COPY(ScopedLogStorer);

    // Save a log message
    void operator()(LogProvenance, LogLevel lev, std::string msg);

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

    //! Get the minimum level being recorded
    LogLevel level() const { return min_level_; }

    //! Change the level to record
    void level(LogLevel lev) { min_level_ = lev; }

    //! Get the minimum level being recorded
    int float_digits() const { return float_digits_; }

    //! Change the level to record
    void float_digits(int fd)
    {
        CELER_EXPECT(fd >= 0);
        float_digits_ = fd;
    }

  private:
    Logger* logger_{nullptr};
    std::unique_ptr<Logger> saved_logger_;
    LogLevel min_level_;
    int float_digits_{4};
    VecString messages_;
    VecString levels_;
};

// Print expected results
std::ostream& operator<<(std::ostream& os, ScopedLogStorer const& logs);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
