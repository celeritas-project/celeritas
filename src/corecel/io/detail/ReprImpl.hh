//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <ostream>
#include <string_view>

#include "corecel/Assert.hh"

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

template<class T, class Enable = void>
struct ReprFormatter
{
    std::ostream& operator()(std::ostream& os, T const& obj) const
    {
        return os << obj;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Write a ReprValue to a stream using its formatter policy.
 */
template<class T>
inline std::ostream& operator<<(std::ostream& os, ReprValue<T> const& s)
{
    return ReprFormatter<T>{}(os, s.obj);
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
        auto repr_element = [](auto const& element) {
            using celeritas::repr;
            return repr(element);
        };

        os << '{'
           << Joined{value.begin, value.end, ", ", UnaryToStream{repr_element}};
        if (value.trailing_comma)
            os << ',';
        os << '}';
        return os;
    }
};

//! Stream a string using repr-specific escaping and formatting.
struct ReprString
{
    char const* s;
    std::size_t size;

    static ReprString from_cstring(char const* s);
    friend std::ostream& operator<<(std::ostream& os, ReprString value);
};

//---------------------------------------------------------------------------//
// HELPER FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//
void repr_char(std::ostream& os, char value);

void write_hex_char(std::ostream& os, unsigned char value);

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
