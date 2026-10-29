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

namespace celeritas
{
namespace detail
{
template<class T>
struct Repr;
}

template<class T>
detail::Repr<T> repr(T const& obj);

namespace detail
{
//---------------------------------------------------------------------------//
// STREAMABLE
//---------------------------------------------------------------------------//
/*!
 * Lightweight wrapper for printing an object with repr-specific formatting.
 */
template<class T>
struct Repr
{
    T const& obj;
};

//---------------------------------------------------------------------------//
/*!
 * Write a streamable object to a stream.
 */
template<class T>
inline std::ostream& operator<<(std::ostream& os, Repr<T> const& s)
{
    ScopedStreamFormat save_fmt(&os);
    return os << s.obj;
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

void write_string_repr(std::ostream& os, std::string_view value);

//---------------------------------------------------------------------------//
}  // namespace detail

template<class T>
detail::Repr<T> repr(T const& obj)
{
    return {obj};
}

}  // namespace celeritas
