//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ReprFormatter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "corecel/Types.hh"  // IWYU pragma: keep

namespace celeritas
{
//! Default formatter that streams values directly.
template<class T, class Enable>
struct ReprFormatter
{
    std::ostream& operator()(std::ostream& os, T const& obj) const
    {
        return os << obj;
    }
};

}  // namespace celeritas
