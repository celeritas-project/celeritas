//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/LoggerMessage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <memory>
#include <utility>

#include "../LoggerTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Stream-like helper class for writing log messages.
 *
 * This class should only be created by a Logger instance. When it destructs,
 * the handler is called to print the information.
 */
class LoggerMessage
{
  public:
    //!@{
    //! \name Type aliases
    using StreamManip = std::ostream& (*)(std::ostream&);
    //!@}

  public:
    // Construct with reference to function object, etc.
    LoggerMessage(LogHandler* handle, Provenance prov, LogLevel lev);

    // Flush message on destruction
    ~LoggerMessage();

    //!@{
    //! Prevent copying but allow moving
    LoggerMessage(LoggerMessage const&) = delete;
    LoggerMessage& operator=(LoggerMessage const&) = delete;
    LoggerMessage(LoggerMessage&&) = default;
    LoggerMessage& operator=(LoggerMessage&&) = default;
    //!@}

    // Write the object to the stream if applicable
    template<class T>
    inline LoggerMessage& operator<<(T&& rhs);

    // Accept manipulators such as std::endl, std::setw
    inline LoggerMessage& operator<<(StreamManip manip);

    // Update the steam state
    inline void setstate(std::ostream::iostate state);

  private:
    LogHandler* handle_;
    Provenance prov_;
    LogLevel lev_;
    std::unique_ptr<std::ostream> os_;
};

//---------------------------------------------------------------------------//
/*!
 * Write the object to the stream if applicable.
 */
template<class T>
LoggerMessage& LoggerMessage::operator<<(T&& rhs)
{
    if (os_)
    {
        *os_ << std::forward<T>(rhs);
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Accept a stream manipulator.
 */
LoggerMessage& LoggerMessage::operator<<(StreamManip manip)
{
    if (os_)
    {
        manip(*os_);
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Update the steam state (needed by some manipulators).
 */
void LoggerMessage::setstate(std::ostream::iostate state)
{
    if (os_)
    {
        os_->setstate(state);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
