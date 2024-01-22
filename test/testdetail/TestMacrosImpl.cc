//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosImpl.cc
//---------------------------------------------------------------------------//
#include "TestMacrosImpl.hh"

#include <cstdio>
#include <cstring>
#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

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
 *
 * \todo for now this just does string equality, but could do a recursive
 * visitor to compare actual values, and do soft equivalence for floating
 * points.
 */
::testing::AssertionResult IsJsonEq(char const*,
                                    char const*,
                                    [[maybe_unused]] std::string_view expected,
                                    [[maybe_unused]] std::string_view actual)
{
#if CELERITAS_USE_JSON
    if (expected == actual)
    {
        return ::testing::AssertionSuccess();
    }

    auto result = ::testing::AssertionFailure();
    result << "Expected:\n  R\"json(" << expected << ")json\"";
    result << "\nActual:\n  R\"json(" << actual << ")json\"";
    return result;
#else
    auto result = ::testing::AssertionFailure();
    result << "JSON is not enabled: wrap this test in 'if "
              "(CELERITAS_USE_JSON)'";
    return result;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
