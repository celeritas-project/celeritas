//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/inp/DistributionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Distributions.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

template<class T>
void to_json(nlohmann::json& j, DeltaDistribution<T> const&);
template<class T>
void from_json(nlohmann::json const& j, DeltaDistribution<T>&);

void to_json(nlohmann::json& j, NormalDistribution const&);
void from_json(nlohmann::json const& j, NormalDistribution&);

void to_json(nlohmann::json& j, IsotropicDistribution const&);
void from_json(nlohmann::json const& j, IsotropicDistribution&);

void to_json(nlohmann::json& j, UniformBoxDistribution const&);
void from_json(nlohmann::json const& j, UniformBoxDistribution&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
