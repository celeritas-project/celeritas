//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Array.hh"
#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read an array from a JSON file.
 */
template<class T, size_type N>
void from_json(const nlohmann::json& j, Array<T, N>& value)
{
    CELER_VALIDATE(j.size() == N,
                   << "unexpected array size (" << j.size()
                   << " in JSON input: should have " << N);
    for (size_type i = 0; i != N; ++i)
    {
        value[i] = j[i].get<T>();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write an array to a JSON file.
 */
template<class T, size_type N>
void to_json(nlohmann::json& j, const Array<T, N>& value)
{
    j = nlohmann::json::array();
    for (size_type i = 0; i != N; ++i)
    {
        j.push_back(value[i]);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
