//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.cc
//---------------------------------------------------------------------------//
#include "ReprImpl.hh"

#include <cctype>
#include <cstdio>

#include "corecel/Assert.hh"

#include "../Repr.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Whether all characters are printable (no escaping needed).
 */
bool all_printable(std::string_view s)
{
    return std::all_of(s.begin(), s.end(), [](unsigned char ch) {
        return std::isprint(ch) || ch == '\n';
    });
}

//---------------------------------------------------------------------------//
/*!
 * Print a character as a hex representation.
 *
 * There are undoubtedly helper libraries that do better than this...
 */
void repr_char(std::ostream& os, char value)
{
    if (std::isprint(value))
    {
        os << value;
    }
    else
    {
        os << '\\';
        switch (value)
        {
            case '\0':
                os << '0';
                break;
            case '\a':
                os << 'a';
                break;
            case '\b':
                os << 'b';
                break;
            case '\t':
                os << 't';
                break;
            case '\n':
                os << 'n';
                break;
            case '\r':
                os << 'r';
                break;
            default:
                os << 'x'
                   << char_to_hex_string(static_cast<unsigned char>(value));
                break;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get a character as a two-digit hexadecimal like '0a'.
 */
std::string char_to_hex_string(unsigned char value)
{
    char buffer[3];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    int size = std::snprintf(buffer, sizeof(buffer), "%02hhx", value);
    CELER_ENSURE(size == 2);
    return {buffer, buffer + 2};
}

//---------------------------------------------------------------------------//
/*!
 * Print a type string to the stream.
 */
void print_simple_type(std::ostream& os, char const* type, char const* name)
{
    os << type;
    if (name)
    {
        os << ' ' << name;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
