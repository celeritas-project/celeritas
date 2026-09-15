//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/FieldIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Field.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, NoField const&);
void from_json(nlohmann::json const& j, NoField&);

void to_json(nlohmann::json& j, UniformField const&);
void from_json(nlohmann::json const& j, UniformField&);

void to_json(nlohmann::json& j, CylMapField const&);
void from_json(nlohmann::json const& j, CylMapField&);

template<class T>
void to_json(nlohmann::json& j, AxisGrid<T> const&);
template<class T>
void from_json(nlohmann::json const& j, AxisGrid<T>&);

void to_json(nlohmann::json& j, CartMapField const&);
void from_json(nlohmann::json const& j, CartMapField&);

void to_json(nlohmann::json& j, Field const&);
void from_json(nlohmann::json const& j, Field&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
