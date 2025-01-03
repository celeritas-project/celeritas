//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/NullLoggerMessage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Stream-like helper class that \em discards everything passed to it.
 *
 * This helper class should simply eat any messages and objects passed to it.
 */
class NullLoggerMessage
{
  public:
    //!@{
    //! \name Type aliases
    using StreamManip = std::ostream& (*)(std::ostream&);
    //!@}

  public:
    // Default constructor.
    NullLoggerMessage() = default;

    //! Do not print this object
    template<class T>
    CELER_FORCEINLINE_FUNCTION NullLoggerMessage& operator<<(T&&)
    {
        return *this;
    }

    //! Ignore this manipulator function
    CELER_FORCEINLINE_FUNCTION NullLoggerMessage& operator<<(StreamManip)
    {
        return *this;
    }

    //! Do not set any state
    CELER_FORCEINLINE_FUNCTION void setstate(std::ostream::iostate) {}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
