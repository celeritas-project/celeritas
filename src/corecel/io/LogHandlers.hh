//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogHandlers.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>

#include "LoggerTypes.hh"

namespace celeritas
{
class MpiCommunicator;

//---------------------------------------------------------------------------//
/*!
 * Simple log handler: write with colors to a long-lived ostream reference.
 */
class StreamLogHandler
{
  public:
    //! Construct with long-lived reference to a stream
    explicit StreamLogHandler(std::ostream& os) : os_{os} {}

    // Write the message to the stream
    void operator()(LogProvenance, LogLevel, std::string) const;

  private:
    std::ostream& os_;
};

//---------------------------------------------------------------------------//
/*!
 * Log with a shared mutex guarding the stream output.
 */
class MutexedStreamLogHandler
{
  public:
    //! Construct with long-lived reference to a stream
    explicit MutexedStreamLogHandler(std::ostream& os) : os_{os} {}

    // Write the message to the stream
    void operator()(LogProvenance, LogLevel, std::string) const;

  private:
    std::ostream& os_;
};

//---------------------------------------------------------------------------//
//! Log the local node number as well as the message
class LocalMpiHandler
{
  public:
    // Construct with long-lived reference to a stream
    LocalMpiHandler(std::ostream& os, MpiCommunicator const& comm);

    // Write with processor ID
    void operator()(LogProvenance prov, LogLevel lev, std::string msg) const;

  private:
    std::ostream& os_;
    int rank_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
