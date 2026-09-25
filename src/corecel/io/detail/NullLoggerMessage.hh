//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/NullLoggerMessage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ios>
#include <iosfwd>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Stream-like helper class that \em discards everything passed to it.
 */
class NullLoggerMessage
{
  public:
    //!@{
    //! \name Type aliases
    using StreamManip = std::ostream& (*)(std::ostream&);
    using IoState = std::ios_base::iostate;
    //!@}

  public:
    //! Do not print this object
    template<class T>
    CELER_CONSTEXPR_FUNCTION NullLoggerMessage& operator<<(T&&)
    {
        return *this;
    }

    //! Ignore this manipulator function
    CELER_CONSTEXPR_FUNCTION NullLoggerMessage& operator<<(StreamManip)
    {
        return *this;
    }

    //! Do not set any state
    CELER_CONSTEXPR_FUNCTION void setstate(IoState) {}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
