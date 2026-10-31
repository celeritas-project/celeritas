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

#include "corecel/io/ScopedStreamFormat.hh"

#include "detail/ReprImpl.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<>
struct ReprFormatter<float>
{
    std::ostream& operator()(std::ostream& os, float obj) const
    {
        ScopedStreamFormat save_fmt(&os);
        os.precision(7);
        return os << obj << 'f';
    }
};

template<>
struct ReprFormatter<double>
{
    std::ostream& operator()(std::ostream& os, double obj) const
    {
        ScopedStreamFormat save_fmt(&os);
        os.precision(14);
        return os << obj;
    }
};

template<>
struct ReprFormatter<char>
{
    std::ostream& operator()(std::ostream& os, char obj) const
    {
        ScopedStreamFormat save_fmt(&os);
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
        return os << "'\\x" << char_to_hex_string(obj) << '\'';
    }
};

template<>
struct ReprFormatter<bool>
{
    std::ostream& operator()(std::ostream& os, bool obj) const
    {
        return os << std::boolalpha << obj;
    }
};

template<>
struct ReprFormatter<unsigned int>
{
    std::ostream& operator()(std::ostream& os, unsigned int obj) const
    {
        return os << obj << 'u';
    }
};

template<>
struct ReprFormatter<long>
{
    std::ostream& operator()(std::ostream& os, long obj) const
    {
        return os << obj << 'l';
    }
};

template<>
struct ReprFormatter<unsigned long>
{
    std::ostream& operator()(std::ostream& os, unsigned long obj) const
    {
        return os << obj << "ul";
    }
};

template<>
struct ReprFormatter<long long>
{
    std::ostream& operator()(std::ostream& os, long long obj) const
    {
        return os << obj << "ll";
    }
};

template<>
struct ReprFormatter<unsigned long long>
{
    std::ostream& operator()(std::ostream& os, unsigned long long obj) const
    {
        return os << obj << "ull";
    }
};

template<>
struct ReprFormatter<char const*>
{
    std::ostream& operator()(std::ostream& os, char const* obj) const
    {
        if (obj)
        {
            write_string_repr(os, obj);
            return os;
        }
        return os << "nullptr";
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
        write_string_repr(os, obj);
        return os;
    }
};

template<std::size_t N>
struct ReprFormatter<char const[N]>
{
    std::ostream& operator()(std::ostream& os, char const (&obj)[N]) const
    {
        write_string_repr(os, obj);
        return os;
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
