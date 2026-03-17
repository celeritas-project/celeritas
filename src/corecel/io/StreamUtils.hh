//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <sstream>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return as a string any object that has an ostream operator.
 */
template<class T>
inline std::string stream_to_string(T const& item)
{
    std::ostringstream os;
    os << item;
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
/*!
 * Write a `to_cstring`-enabled enum to a stream.
 */
template<class T>
auto operator<<(std::ostream& os, T value) -> std::enable_if_t<
    std::is_enum_v<T> && std::is_same_v<decltype(to_cstring(value)), char const*>,
    std::ostream&>
{
    return os << to_cstring(value);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
