//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
 * Whether the string starts with another string.
 */
bool starts_with(std::string const& main_string, std::string const& prefix)
{
    if (main_string.size() < prefix.size())
        return false;

    return std::equal(main_string.begin(),
                      main_string.begin() + prefix.size(),
                      prefix.begin());
}

//---------------------------------------------------------------------------//
/*!
 * Whether the string ends with another string.
 */
bool ends_with(std::string const& main_string, std::string const& suffix)
{
    if (main_string.size() < suffix.size())
        return false;

    return std::equal(
        main_string.end() - suffix.size(), main_string.end(), suffix.begin());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
