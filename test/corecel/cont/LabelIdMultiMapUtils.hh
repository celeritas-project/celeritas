//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LabelIdMultiMapUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/cont/LabelIdMultiMap.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class I>
inline auto get_multimap_labels(LabelIdMultiMap<I> const& multimap)
{
    std::vector<std::string> result;
    for (auto id : range(id_cast<I>(multimap.size())))
    {
        result.push_back(to_string(multimap.at(id)));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
