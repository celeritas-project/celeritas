//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/SpanIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a span to a JSON file.
 */
template<class T, std::size_t N>
void to_json(nlohmann::json& j, Span<T, N> const& value)
{
    j = nlohmann::json::array();
    for (std::size_t i = 0; i != value.size(); ++i)
    {
        j.push_back(value[i]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
