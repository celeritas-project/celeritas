//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypesIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

template<class T>
void from_json(nlohmann::json const& j, Tolerance<T>& value);
template<class T>
void to_json(nlohmann::json& j, Tolerance<T> const& value);

extern template void from_json(nlohmann::json const&, Tolerance<real_type>&);
extern template void to_json(nlohmann::json&, Tolerance<real_type> const&);

void from_json(nlohmann::json const& j, LogicNotation& value);
void to_json(nlohmann::json& j, LogicNotation const& value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
