//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/io/StreamToString.hh"

#include "../ScopedStreamFormat.hh"

namespace celeritas
{
template<class T>
struct ReprTraits;

namespace detail
{
//---------------------------------------------------------------------------//
// STREAMABLE
//---------------------------------------------------------------------------//
/*!
 * Thin temporary wrapper for printing a complex class to a stream.
 */
template<class T>
struct Repr
{
    T const& obj;
    char const* name = nullptr;
};

//---------------------------------------------------------------------------//
/*!
 * Write a streamable object to a stream.
 */
template<class T>
inline std::ostream& operator<<(std::ostream& os, Repr<T> const& s)
{
    ScopedStreamFormat save_fmt(&os);
    ReprTraits<T>::init(os);
    if (s.name)
    {
        ReprTraits<T>::print_type(os, s.name);
        os << '{';
    }
    ReprTraits<T>::print_value(os, s.obj);
    if (s.name)
    {
        os << '}';
    }
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a streamable object to a string.
 */
template<class T>
inline std::string to_string(Repr<T> const& s)
{
    return stream_to_string(s);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//
bool all_printable(std::string_view s);

void repr_char(std::ostream& os, char value);

std::string char_to_hex_string(unsigned char value);

void print_simple_type(std::ostream& os, char const* type, char const* name);

template<class T>
inline void
print_container_type(std::ostream& os, char const* type, char const* name)
{
    os << type << '<';
    ReprTraits<T>::print_type(os);
    os << '>';
    if (name)
    {
        os << ' ' << name;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
