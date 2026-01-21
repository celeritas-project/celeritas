//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/LoggerMessage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <sstream>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/io/LoggerTypes.hh"

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
    using StreamManip = std::ios_base& (*)(std::ios_base&);
    //!@}

  public:
    // Construct with reference to function object, etc.
    inline LoggerMessage(LogHandler const* handle,
                         LogProvenance&& prov,
                         LogLevel lev);

    // Flush message on destruction
    inline ~LoggerMessage();

    //! Prevent copying but allow moving
    CELER_DEFAULT_MOVE_DELETE_COPY(LoggerMessage);

    // Write the object to the stream if applicable
    template<class T>
    inline LoggerMessage& operator<<(T&& rhs);

    // Accept manipulators such as std::endl, std::setw
    inline LoggerMessage& operator<<(StreamManip manip);

    // Update the stream state
    inline void setstate(std::ostream::iostate state);

  private:
    LogHandler const* handle_;
    LogProvenance prov_;
    LogLevel lev_;
    std::unique_ptr<std::ostringstream> os_;

    // Create the message when handle is non-null
    void construct_impl(LogProvenance&& prov, LogLevel lev);

    // Flush the message during destruction
    void destroy_impl() noexcept;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with reference to function object, etc.
 *
 * The handle *may be* null, indicating that the output of this message will
 * not be displayed.
 */
CELER_FORCEINLINE LoggerMessage::LoggerMessage(LogHandler const* handle,
                                               LogProvenance&& prov,
                                               LogLevel lev)
    : handle_(handle)
{
    if (CELER_UNLIKELY(handle_))
    {
        this->construct_impl(std::move(prov), lev);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Flush message on destruction.
 *
 * Note that due to move semantics, it's possible for a stale "moved"
 * LoggerMessage to have a nonnull \c handle_ but a null \c os_ .
 */
CELER_FORCEINLINE LoggerMessage::~LoggerMessage()
{
    if (CELER_UNLIKELY(os_))
    {
        this->destroy_impl();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write the object to the stream if applicable.
 */
template<class T>
CELER_FORCEINLINE LoggerMessage& LoggerMessage::operator<<(T&& rhs)
{
    if (CELER_UNLIKELY(os_))
    {
        *os_ << std::forward<T>(rhs);
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Accept a stream manipulator.
 */
CELER_FORCEINLINE LoggerMessage& LoggerMessage::operator<<(StreamManip manip)
{
    if (CELER_UNLIKELY(os_))
    {
        manip(*os_);
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Update the steam state (needed by some manipulators).
 */
CELER_FORCEINLINE void LoggerMessage::setstate(std::ostream::iostate state)
{
    if (CELER_UNLIKELY(os_))
    {
        os_->setstate(state);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
