//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <cstdio>
#include <cstring>
#include <string>
#include "base/Assert.hh"
#include "base/ColorUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Get the "skip" message for the skip macro.
 */
const char* skip_cstring()
{
    static const std::string str = std::string(color_code('y'))
                                   + std::string("[   SKIP   ]")
                                   + std::string(color_code('d'));
    return str.c_str();
}

//---------------------------------------------------------------------------//
/*!
 * Number of base-10 digits in an unsigned integer.
 *
 * This function is useful for pre-calculating field widths for printing.
 */
unsigned int num_digits(unsigned int val)
{
    if (val == 0)
        return 1;

    unsigned int result = 0;
    unsigned int cur    = 1;

    while (cur <= val)
    {
        cur *= 10;
        ++result;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a character as a two-digit hexadecimal like '0a'.
 */
std::string char_to_hex_string(unsigned char value)
{
    char buffer[8];
    std::sprintf(buffer, "%02hhx", value);
    return {buffer, buffer + 2};
}

//---------------------------------------------------------------------------//
/*!
 * Return a replacement string if the given string is too long.
 *
 * where too long means > digits digits.
 */
const char*
trunc_string(unsigned int digits, const char* str, const char* trunc)
{
    CELER_EXPECT(str && trunc);
    CELER_EXPECT(digits > 0);
    CELER_EXPECT(std::strlen(trunc) <= digits);

    if (std::strlen(str) > digits)
    {
        return trunc;
    }
    return str;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
