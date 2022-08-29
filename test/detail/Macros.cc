//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/Macros.cc
//---------------------------------------------------------------------------//
#include "Macros.hh"

#include <cstdio>
#include <cstring>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"

namespace celeritas
{
namespace test
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Number of base-10 digits in an unsigned integer.
 *
 * This function is useful for pre-calculating field widths for printing.
 */
int num_digits(unsigned long val)
{
    if (val == 0)
        return 1;

    int           result = 0;
    unsigned long cur    = 1;

    while (cur <= val)
    {
        cur *= 10;
        ++result;
    }

    return result;
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
} // namespace test
} // namespace celeritas
