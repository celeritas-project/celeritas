//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ReprStd.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "detail/ReprImpl.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
inline std::ostream& operator<<(std::ostream& os,
                                Repr<std::string_view> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                Repr<std::string> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

template<class T1, class T2>
std::ostream& operator<<(std::ostream& os, Repr<std::pair<T1, T2>> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os << '{' << repr(value.obj.first) << ',' << repr(value.obj.second) << '}';
    return os;
}

template<class T, class A>
std::ostream& operator<<(std::ostream& os, Repr<std::vector<T, A>> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os << '{';
    std::size_t count = 0;
    for (auto const& item : value.obj)
    {
        if (count++ != 0)
        {
            os << ", ";
        }
        os << repr(item);
    }
    if (value.obj.size() > 8)
    {
        os << ',';
    }
    return os << '}';
}

template<class T, class C, class A>
std::ostream& operator<<(std::ostream& os, Repr<std::set<T, C, A>> const& value)
{
    ScopedStreamFormat save_fmt(&os);
    os << '{';
    bool first = true;
    for (auto const& item : value.obj)
    {
        if (!first)
        {
            os << ", ";
        }
        first = false;
        os << repr(item);
    }
    return os << '}';
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
