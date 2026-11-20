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

#include "ReprFormatter.hh"

#include "detail/ReprContainer.hh"
#include "detail/ReprImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<>
struct ReprFormatter<std::string_view>
{
    std::ostream& operator()(std::ostream& os, std::string_view obj) const
    {
        return os << detail::ReprString{obj.data(), obj.size()};
    }
};

template<>
struct ReprFormatter<std::string>
{
    std::ostream& operator()(std::ostream& os, std::string const& obj) const
    {
        return os << detail::ReprString{obj.data(), obj.size()};
    }
};

template<class T1, class T2>
struct ReprFormatter<std::pair<T1, T2>>
{
    std::ostream& operator()(std::ostream& os,
                             std::pair<T1, T2> const& obj) const
    {
        os << '{' << repr(obj.first) << ',' << repr(obj.second) << '}';
        return os;
    }
};

template<class T, class A>
struct ReprFormatter<std::vector<T, A>>
{
    std::ostream& operator()(std::ostream& os,
                             std::vector<T, A> const& vec) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return os << detail::ReprContainer{
                       vec.begin(), vec.end(), vec.size() > 8};
        }
        else
        {
            auto const* data = vec.data();
            return os << detail::ReprContainer{
                       data, data + vec.size(), vec.size() > 8};
        }
    }
};

template<class T, class C, class A>
struct ReprFormatter<std::set<T, C, A>>
{
    std::ostream& operator()(std::ostream& os,
                             std::set<T, C, A> const& obj) const
    {
        return os << detail::ReprContainer{
                   obj.begin(), obj.end(), obj.size() > 8};
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
