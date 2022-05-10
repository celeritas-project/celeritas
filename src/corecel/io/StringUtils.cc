//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringUtils.cc
//---------------------------------------------------------------------------//
#include "StringUtils.hh"

#include <algorithm>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether the string ends with another string.
 */
bool ends_with(const std::string& main_string, const std::string& suffix)
{
    if (main_string.size() < suffix.size())
        return false;

    return std::equal(
        main_string.end() - suffix.size(), main_string.end(), suffix.begin());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
