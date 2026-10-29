//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include "corecel/Assert.hh"
#include "corecel/io/ScopedStreamFormat.hh"
#include "corecel/io/StreamUtils.hh"

#include "Joined.hh"

namespace celeritas
{
namespace detail
{
template<class T>
struct ReprValue;
}

template<class T>
detail::ReprValue<T> repr(T const& obj);

namespace detail
{
//---------------------------------------------------------------------------//
// STREAMABLE
//---------------------------------------------------------------------------//

/*!
 * Lightweight wrapper for printing an object with repr-specific formatting.
 */
template<class T>
struct ReprValue
{
    T const& obj;
};

//---------------------------------------------------------------------------//
/*!
 * Write a streamable object to a stream.
 *
 * This must be a free function so it can be overridden by std types.
 */
template<class T>
inline std::ostream& operator<<(std::ostream& os, ReprValue<T> const& s)
{
    return os << s.obj;
}

//! Implementation for ReprContainer
template<class T>
void generic_repr(std::ostream& os, T const& value)
{
    using celeritas::repr;
    os << repr(value);
}

//! Stream a range using repr for each element.
template<class Iter>
struct ReprContainer
{
    Iter begin;
    Iter end;
    bool trailing_comma{false};

    friend std::ostream& operator<<(std::ostream& os,
                                    ReprContainer const& value)
    {
        using ValueT = decltype(*value.begin);

        os << '{'
           << Joined{value.begin, value.end, ", ", &generic_repr<ValueT>};
        if (value.trailing_comma)
            os << ',';
        os << '}';
        return os;
    }
};
//---------------------------------------------------------------------------//
// HELPER FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//
bool all_printable(std::string_view s);

void repr_char(std::ostream& os, char value);

std::string char_to_hex_string(unsigned char value);

void write_string_repr(std::ostream& os, std::string_view value);

//---------------------------------------------------------------------------//
}  // namespace detail

//! Default "repr" allows overriding via detail class.
template<class T>
detail::ReprValue<T> repr(T const& obj)
{
    return {obj};
}

}  // namespace celeritas
