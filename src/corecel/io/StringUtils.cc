//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringUtils.cc
//---------------------------------------------------------------------------//
#include "StringUtils.hh"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether the string starts with another string.
 */
bool starts_with(std::string_view main_string, std::string_view prefix)
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
bool ends_with(std::string_view main_string, std::string_view suffix)
{
    if (main_string.size() < suffix.size())
        return false;

    return std::equal(
        main_string.end() - suffix.size(), main_string.end(), suffix.begin());
}

//---------------------------------------------------------------------------//
/*!
 * Whether the character is whitespace or unprintable.
 */
bool is_ignored_trailing(unsigned char c)
{
    return std::isspace(c) || !std::isprint(c);
}

//---------------------------------------------------------------------------//
/*!
 * Test C strings for equality, allowing one or the other to be null.
 *
 * If one pointer is null, the result compares \c false.
 */
bool cstring_equal(char const* lhs, char const* rhs)
{
    CELER_EXPECT(rhs || lhs);
    if (!lhs || !rhs)
        return false;
    return std::strcmp(lhs, rhs) == 0;
}

//---------------------------------------------------------------------------//
/*!
 * Return a string view with leading and trailing whitespace removed.
 */
std::string_view trim(std::string_view input)
{
    auto start = input.begin();
    auto stop = input.end();
    while (start != stop && is_ignored_trailing(*start))
    {
        ++start;
    }
    while (start != stop && is_ignored_trailing(*(stop - 1)))
    {
        --stop;
    }
    return {&(*start), static_cast<std::size_t>(stop - start)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
