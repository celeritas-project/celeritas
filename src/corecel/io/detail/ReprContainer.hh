//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprContainer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Joined.hh"
#include "ReprFwd.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
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

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
