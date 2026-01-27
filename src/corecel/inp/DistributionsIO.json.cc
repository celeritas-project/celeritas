//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/inp/DistributionsIO.json.cc
//---------------------------------------------------------------------------//
#include "DistributionsIO.json.hh"

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

template<class T>
void to_json(nlohmann::json& j, DeltaDistribution<T> const& v)
{
    j = nlohmann::json{{"delta", v.value}};
}

template<class T>
void from_json(nlohmann::json const& j, DeltaDistribution<T>& v)
{
    j.at("delta").get_to(v.value);
}

void to_json(nlohmann::json& j, NormalDistribution const& v)
{
    j = nlohmann::json{
        {"normal", {{"mean", v.mean}, {"stddev", v.stddev}}},
    };
}

void from_json(nlohmann::json const& j, NormalDistribution& v)
{
    auto const& params = j.at("normal");
    params.at("mean").get_to(v.mean);
    params.at("stddev").get_to(v.stddev);
}

void to_json(nlohmann::json& j, IsotropicDistribution const&)
{
    j = nlohmann::json{{"isotropic", nlohmann::json::object()}};
}

void from_json(nlohmann::json const&, IsotropicDistribution&) {}

void to_json(nlohmann::json& j, UniformBoxDistribution const& v)
{
    j = nlohmann::json{
        {"uniform_box", {{"lower", v.lower}, {"upper", v.upper}}},
    };
}

void from_json(nlohmann::json const& j, UniformBoxDistribution& v)
{
    auto const& params = j.at("uniform_box");
    params.at("lower").get_to(v.lower);
    params.at("upper").get_to(v.upper);
}

//!@}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

template void from_json(nlohmann::json const&, DeltaDistribution<double>&);
template void to_json(nlohmann::json&, DeltaDistribution<double> const&);
template void
from_json(nlohmann::json const&, DeltaDistribution<Array<double, 3>>&);
template void
to_json(nlohmann::json&, DeltaDistribution<Array<double, 3>> const&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
