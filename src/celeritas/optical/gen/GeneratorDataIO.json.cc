//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorDataIO.json.cc
//---------------------------------------------------------------------------//
#include "GeneratorDataIO.json.hh"

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/math/QuantityIO.json.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, GeneratorStepData const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, speed),
        CELER_JSON_PAIR(v, time),
        CELER_JSON_PAIR(v, pos),
    };
}

void from_json(nlohmann::json const& j, GeneratorStepData& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, speed);
    CELER_JSON_LOAD_REQUIRED(j, v, time);
    CELER_JSON_LOAD_REQUIRED(j, v, pos);
}

void to_json(nlohmann::json& j, GeneratorDistributionData const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, type),
        CELER_JSON_PAIR(v, num_photons),
        CELER_JSON_PAIR(v, primary),
        CELER_JSON_PAIR(v, step_length),
        CELER_JSON_PAIR(v, charge),
        CELER_JSON_PAIR(v, material),
        CELER_JSON_PAIR(v, continuous_edep_fraction),
        {"points",
         {
             {"pre", v.points[StepPoint::pre]},
             {"post", v.points[StepPoint::post]},
         }},
    };
}

void from_json(nlohmann::json const& j, GeneratorDistributionData& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, type);
    CELER_JSON_LOAD_REQUIRED(j, v, num_photons);
    CELER_JSON_LOAD_REQUIRED(j, v, primary);
    CELER_JSON_LOAD_REQUIRED(j, v, step_length);
    CELER_JSON_LOAD_REQUIRED(j, v, charge);
    CELER_JSON_LOAD_REQUIRED(j, v, material);
    CELER_JSON_LOAD_REQUIRED(j, v, continuous_edep_fraction);
    auto const& points = j.at("points");
    points.at("pre").get_to(v.points[StepPoint::pre]);
    points.at("post").get_to(v.points[StepPoint::post]);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
