//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/EnumBitsetIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "EnumBitset.hh"
#include "Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write an enum bitset to a JSON file.
 *
 * \note A \c to_cstring function for the underlying type must be defined.
 */
template<class E>
void to_json(nlohmann::json& j, EnumBitset<E> const& v)
{
    j = nlohmann::json::array();
    for (auto enum_val : range(E::size_))
    {
        if (v[enum_val])
        {
            j.push_back(to_cstring(enum_val));
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
