//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/StringSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Remove platform-sensitive components from strings to improve testability.
 *
 * - Removes pointers
 * - Removes ANSI escape sequences
 * - Rounds floating points to a given digit of precision
 */
class StringSimplifier
{
  public:
    // Construct with number of digit of precision
    inline StringSimplifier(int precision);
    StringSimplifier() = default;

    // Simplify
    [[nodiscard]] std::string operator()(std::string const& old) const;

  private:
    int precision_{4};

    std::string simplify_sci(std::string s) const;
    std::string simplify_float(std::string s) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
StringSimplifier::StringSimplifier(int precision) : precision_{precision}
{
    CELER_EXPECT(precision_ > 0);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
