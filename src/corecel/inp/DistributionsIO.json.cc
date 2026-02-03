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
    j = {json_type_pair("delta"), CELER_JSON_PAIR(v, value)};
}

template<class T>
void from_json(nlohmann::json const& j, DeltaDistribution<T>& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, value);
}

void to_json(nlohmann::json& j, NormalDistribution const& v)
{
    j = {
        json_type_pair("normal"),
        CELER_JSON_PAIR(v, mean),
        CELER_JSON_PAIR(v, stddev),
    };
}

void from_json(nlohmann::json const& j, NormalDistribution& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, mean);
    CELER_JSON_LOAD_REQUIRED(j, v, stddev);
}

void to_json(nlohmann::json& j, IsotropicDistribution const&)
{
    j = {json_type_pair("isotropic")};
}

void from_json(nlohmann::json const&, IsotropicDistribution&) {}

void to_json(nlohmann::json& j, UniformBoxDistribution const& v)
{
    j = {
        json_type_pair("uniform_box"),
        CELER_JSON_PAIR(v, lower),
        CELER_JSON_PAIR(v, upper),
    };
}

void from_json(nlohmann::json const& j, UniformBoxDistribution& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, lower);
    CELER_JSON_LOAD_REQUIRED(j, v, upper);
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
