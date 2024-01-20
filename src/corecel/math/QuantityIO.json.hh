//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/QuantityIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"

#include "Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read a quantity from a JSON file.
 */
template<class UnitT, class ValueT>
void from_json(nlohmann::json const& j, Quantity<UnitT, ValueT>& q)
{
    static_assert(sizeof(UnitT::label()) > 0,
                  "Unit does not have a 'label' definition");
    if (j.is_array())
    {
        CELER_VALIDATE(j.size() == 2,
                       << "unexpected array size (" << j.size()
                       << ") for quantity in JSON input: should be [value, "
                          "\"units\"]");
        CELER_VALIDATE(j[1].get<std::string>() == UnitT::label(),
                       << "incorrect units '" << j[1].get<std::string>()
                       << "' in JSON input: expected '" << UnitT::label()
                       << "'");
        q = Quantity<UnitT, ValueT>{j[0].get<ValueT>()};
    }
    else
    {
        q = Quantity<UnitT, ValueT>{j.get<ValueT>()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a quantity to a JSON file.
 */
template<class UnitT, class ValueT>
void to_json(nlohmann::json& j, Quantity<UnitT, ValueT> const& q)
{
    static_assert(sizeof(UnitT::label()) > 0,
                  "Unit does not have a 'label' definition");
    j = {q.value(), UnitT::label()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
