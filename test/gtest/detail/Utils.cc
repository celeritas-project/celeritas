//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <string>
#include "base/ColorUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * \brief Get the "skip" message for the skip macro
 */
const char* skip_cstring()
{
    static const std::string str = std::string(color_code('y'))
                                   + std::string("[   SKIP   ]")
                                   + std::string(color_code('d'));
    return str.c_str();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
