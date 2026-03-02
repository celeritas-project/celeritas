//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamToString.hh
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
}  // namespace celeritas
