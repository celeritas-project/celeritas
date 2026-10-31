//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ReprBuiltin.hh
//! \sa corecel/io/Repr.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <ostream>
#include <type_traits>

#include "corecel/io/ScopedStreamFormat.hh"

#include "detail/ReprImpl.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// FLOATING POINT
//---------------------------------------------------------------------------//

template<class T>
struct ReprFormatter<T, std::enable_if_t<std::is_floating_point_v<T>>>
{
    std::ostream& operator()(std::ostream& os, T obj) const
    {
        ScopedStreamFormat save_fmt(&os);
        os.precision(std::is_same_v<T, float>    ? 7
                     : std::is_same_v<T, double> ? 14
                                                 : 16);
        os << obj;

        if constexpr (std::is_same_v<T, float>)
        {
            os << 'f';
        }
        else if constexpr (std::is_same_v<T, long double>)
        {
            os << 'L';
        }
        return os;
    }
};

//---------------------------------------------------------------------------//
// CHARS
//---------------------------------------------------------------------------//

template<>
struct ReprFormatter<char>
{
    std::ostream& operator()(std::ostream& os, char obj) const
    {
        os << '\'';
        repr_char(os, obj);
        return os << '\'';
    }
};

template<>
struct ReprFormatter<unsigned char>
{
    std::ostream& operator()(std::ostream& os, unsigned char obj) const
    {
        os << "'\\x";
        write_hex_char(os, obj);
        os << '\'';
        return os;
    }
};

//---------------------------------------------------------------------------//
// BOOL
//---------------------------------------------------------------------------//

template<>
struct ReprFormatter<bool>
{
    std::ostream& operator()(std::ostream& os, bool obj) const
    {
        return os << std::boolalpha << obj;
    }
};

//---------------------------------------------------------------------------//
// INTEGERS
//---------------------------------------------------------------------------//

template<class T>
struct ReprFormatter<T, std::enable_if_t<std::is_integral_v<T>>>
{
    std::ostream& operator()(std::ostream& os, T obj) const
    {
        os << obj;
        if constexpr (std::is_same_v<T, unsigned int>)
        {
            os << 'u';
        }
        else if constexpr (std::is_same_v<T, long>)
        {
            os << 'l';
        }
        else if constexpr (std::is_same_v<T, unsigned long>)
        {
            os << "ul";
        }
        else if constexpr (std::is_same_v<T, long long>)
        {
            os << "ll";
        }
        else if constexpr (std::is_same_v<T, unsigned long long>)
        {
            os << "ull";
        }
        return os;
    }
};

//---------------------------------------------------------------------------//
// STRINGS
//---------------------------------------------------------------------------//

template<>
struct ReprFormatter<char const*>
{
    std::ostream& operator()(std::ostream& os, char const* obj) const
    {
        if (!obj)
        {
            return os << "nullptr";
        }
        return os << ReprString::from_cstring(obj);
    }
};

template<>
struct ReprFormatter<char*>
{
    std::ostream& operator()(std::ostream& os, char const* obj) const
    {
        return ReprFormatter<char const*>{}(os, obj);
    }
};

template<std::size_t N>
struct ReprFormatter<char[N]>
{
    std::ostream& operator()(std::ostream& os, char const (&obj)[N]) const
    {
        return os << ReprString::from_cstring(obj);
    }
};

template<std::size_t N>
struct ReprFormatter<char const[N]>
{
    std::ostream& operator()(std::ostream& os, char const (&obj)[N]) const
    {
        return os << ReprString::from_cstring(obj);
    }
};

template<class T, std::size_t N>
struct ReprFormatter<T[N]>
{
    std::ostream& operator()(std::ostream& os, T const (&obj)[N]) const
    {
        return os << ReprContainer{obj, obj + N, N > 8};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
