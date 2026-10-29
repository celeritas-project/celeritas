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
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/ReprImpl.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
inline std::ostream& operator<<(std::ostream& os,
                                ReprValue<std::string_view> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                ReprValue<std::string> const& value)
{
    write_string_repr(os, value.obj);
    return os;
}

template<class T1, class T2>
std::ostream& operator<<(std::ostream& os,
                         ReprValue<std::pair<T1, T2>> const& value)
{
    os << '{' << repr(value.obj.first) << ',' << repr(value.obj.second) << '}';
    return os;
}

template<class T, class A>
std::ostream& operator<<(std::ostream& os,
                         ReprValue<std::vector<T, A>> const& value)
{
    auto const& vec = value.obj;
    if constexpr (std::is_same_v<T, bool>)
    {
        return os << ReprContainer{vec.begin(), vec.end(), vec.size() > 8};
    }
    else
    {
        auto const* data = vec.data();
        return os << ReprContainer{data, data + vec.size(), vec.size() > 8};
    }
}

template<class T, class C, class A>
std::ostream& operator<<(std::ostream& os,
                         ReprValue<std::set<T, C, A>> const& value)
{
    auto const& obj = value.obj;
    return os << ReprContainer{obj.begin(), obj.end(), obj.size() > 8};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
