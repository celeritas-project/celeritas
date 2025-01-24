//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringUtils.hh
//! \brief Helper functions for string processing
//---------------------------------------------------------------------------//
#pragma once

#include <cstdlib>  // IWYU pragma: keep
#include <string>
#include <string_view>

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether the string starts with another string.
bool starts_with(std::string_view main_string, std::string_view prefix);

//---------------------------------------------------------------------------//
// Whether the string ends with another string.
bool ends_with(std::string_view main_string, std::string_view suffix);

//---------------------------------------------------------------------------//
// Whether the character is whitespace or unprintable
bool is_ignored_trailing(unsigned char c);

//---------------------------------------------------------------------------//
// Compare two C strings for equality, allowing null for one
bool cstring_equal(char const* lhs, char const* rhs);

//---------------------------------------------------------------------------//
// Return a string view with leading and trailing whitespace removed
[[nodiscard]] std::string_view trim(std::string_view input);

//---------------------------------------------------------------------------//
// Return a lower-cased copy of the input string
[[nodiscard]] std::string tolower(std::string_view input);

//---------------------------------------------------------------------------//
}  // namespace celeritas
