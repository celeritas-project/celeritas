//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ReprBuiltin.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <ostream>
#include <type_traits>

#include "detail/ReprImpl.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
inline std::ostream& operator<<(std::ostream& os, Repr<float> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os.precision(7);
    return os << value.obj << 'f';
}

inline std::ostream& operator<<(std::ostream& os, Repr<double> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os.precision(14);
    return os << value.obj;
}

inline std::ostream& operator<<(std::ostream& os, Repr<char> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os << '\'';
    repr_char(os, value.obj);
    return os << '\'';
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<unsigned char> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    return os << "'\\x" << char_to_hex_string(value.obj) << '\'';
}

inline std::ostream& operator<<(std::ostream& os, Repr<bool> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    return os << std::boolalpha << value.obj;
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<unsigned int> const& value)
{
    return os << value.obj << 'u';
}

inline std::ostream& operator<<(std::ostream& os, Repr<long> const& value)
{
    return os << value.obj << 'l';
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<unsigned long> const& value)
{
    return os << value.obj << "ul";
}

inline std::ostream& operator<<(std::ostream& os, Repr<long long> const& value)
{
    return os << value.obj << "ll";
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<unsigned long long> const& value)
{
    return os << value.obj << "ull";
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<char const*> const& value)
{
    if (value.obj)
    {
        write_string_repr(os, value.obj);
        return os;
    }
    return os << "nullptr";
}

inline std::ostream& operator<<(std::ostream& os, Repr<char*> const& value)
{
    return os << Repr<char const*>{value.obj};
}

template<std::size_t N>
std::ostream& operator<<(std::ostream& os, Repr<char[N]> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

template<std::size_t N>
std::ostream& operator<<(std::ostream& os, Repr<char const[N]> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

template<class T,
         std::size_t N,
         std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, char>, bool> = true>
std::ostream& operator<<(std::ostream& os, Repr<T[N]> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os << '{';
    for (std::size_t i = 0; i < N; ++i)
    {
        if (i != 0)
        {
            os << ", ";
        }
        os << repr(value.obj[i]);
    }
    return os << '}';
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
