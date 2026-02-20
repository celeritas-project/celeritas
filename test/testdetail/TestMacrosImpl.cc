//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosImpl.cc
//---------------------------------------------------------------------------//
#include "TestMacrosImpl.hh"

#include <cstdio>
#include <cstring>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"

#include "JsonComparer.hh"

namespace celeritas
{
namespace testdetail
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

    int result = 0;
    unsigned long cur = 1;

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
char const*
trunc_string(unsigned int digits, char const* str, char const* trunc)
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
/*!
 * Compare two JSON objects.
 */
::testing::AssertionResult IsJsonEq(char const*,
                                    char const*,
                                    std::string_view expected,
                                    std::string_view actual)
{
    JsonComparer compare{};
    auto result = compare(expected, actual);
    if (!result)
    {
        // Print actual result for copy-pasting into "expected" expression
        result << "\n/*** ACTUAL ***/\nR\"json(" << actual
               << ")json\"\n/******/";
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
