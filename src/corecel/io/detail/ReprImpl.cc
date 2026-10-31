//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.cc
//---------------------------------------------------------------------------//
#include "ReprImpl.hh"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "corecel/Assert.hh"
#include "corecel/io/ScopedStreamFormat.hh"

namespace celeritas
{
namespace detail
{
namespace
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
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Write a character as a two-digit hexadecimal like '0a'.
 */
void write_hex_char(std::ostream& os, unsigned char value)
{
    char buffer[3];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    int size = std::snprintf(buffer, sizeof(buffer), "%02hhx", value);
    CELER_ENSURE(size == 2);
    os << buffer;
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
                os << 'x';
                write_hex_char(os, static_cast<unsigned char>(value));
                break;
        }
    }
}
//---------------------------------------------------------------------------//
/*!
 * Write a string using escaped or raw C++ string literal syntax.
 */
void write_string_repr(std::ostream& os, std::string_view value)
{
    ScopedStreamFormat save_fmt(&os);
    using ssize = std::streamsize;
    ssize width = os.width();
    os.width(std::max(width - static_cast<ssize>(value.size()) - 2, ssize{0}));

    if (all_printable(value)
        && (value.size() > 70
            || std::count(value.begin(), value.end(), '"') > 2))
    {
        os << "R\"(" << value << ")\"";
        return;
    }

    os << '"';
    for (char c : value)
    {
        if (c == '"')
        {
            os << '\\';
        }
        repr_char(os, c);
    }
    os << '"';
}

std::ostream& operator<<(std::ostream& os, ReprString const& value)
{
    auto const size = value.size == ReprString::unknown_size
                          ? std::strlen(value.s)
                          : value.size;
    write_string_repr(os, std::string_view{value.s, size});
    return os;
}

}  // namespace detail
}  // namespace celeritas
