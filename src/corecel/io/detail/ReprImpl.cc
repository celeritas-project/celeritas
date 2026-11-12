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
#include <ostream>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
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
bool all_printable(char const* s, std::size_t size)
{
    return std::all_of(s, s + size, [](unsigned char ch) {
        return std::isprint(ch) || ch == '\n';
    });
}

//! Sentinel value for cstrings: address is compared directly
char const null_string_sentinel[] = "nullptr";

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
    os << buffer[0] << buffer[1];
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

ReprString ReprString::from_cstring(char const* s)
{
    if (s == nullptr)
    {
        return ReprString::from_cstring(null_string_sentinel);
    }
    return ReprString{s, std::strlen(s)};
}

//---------------------------------------------------------------------------//
/*!
 * Write a string using escaped or raw C++ string literal syntax.
 */
std::ostream& operator<<(std::ostream& os, ReprString value)
{
    CELER_EXPECT(value.s != nullptr || value.size == 0);

    if (CELER_UNLIKELY(value.s == null_string_sentinel))
    {
        // Print nullptr without surrounding quotes
        return os << null_string_sentinel;
    }

    ScopedStreamFormat save_fmt(&os);
    using ssize = std::streamsize;
    ssize width = os.width();
    os.width(std::max(width - static_cast<ssize>(value.size) - 2, ssize{0}));

    auto const s_end = value.s + value.size;
    if (all_printable(value.s, value.size)
        && (value.size > 70 || std::count(value.s, s_end, '"') > 2))
    {
        os << "R\"(";
        os.write(value.s, value.size);
        os << ")\"";
        return os;
    }

    os << '"';
    for (char const* s = value.s; s != s_end; ++s)
    {
        if (*s == '"')
        {
            os << '\\';
        }
        repr_char(os, *s);
    }
    os << '"';

    return os;
}

}  // namespace detail
}  // namespace celeritas
