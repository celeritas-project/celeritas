//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoIO.cc
//---------------------------------------------------------------------------//
#include "KNDemoIO.hh"

#include <string>
#include <nlohmann/json.hpp>

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, DeviceGridParams const& v)
{
    j = nlohmann::json{{"sync", v.sync}};
}

void from_json(nlohmann::json const& j, DeviceGridParams& v)
{
    j.at("sync").get_to(v.sync);
}

void to_json(nlohmann::json& j, KNDemoRunArgs const& v)
{
    j = nlohmann::json{{"energy", v.energy},
                       {"seed", v.seed},
                       {"num_tracks", v.num_tracks},
                       {"max_steps", v.max_steps},
                       {"tally_grid", v.tally_grid}};
}

void from_json(nlohmann::json const& j, KNDemoRunArgs& v)
{
    j.at("energy").get_to(v.energy);
    j.at("seed").get_to(v.seed);
    j.at("num_tracks").get_to(v.num_tracks);
    j.at("max_steps").get_to(v.max_steps);
    j.at("tally_grid").get_to(v.tally_grid);
}

void to_json(nlohmann::json& j, KNDemoResult const& v)
{
    j = nlohmann::json{{"time", v.time},
                       {"alive", v.alive},
                       {"edep", v.edep},
                       {"total_time", v.total_time}};
}

void from_json(nlohmann::json const& j, KNDemoResult& v)
{
    j.at("time").get_to(v.time);
    j.at("alive").get_to(v.alive);
    j.at("edep").get_to(v.edep);
    j.at("total_time").get_to(v.total_time);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas

namespace celeritas
{
void to_json(nlohmann::json& j, UniformGridData const& v)
{
    j = nlohmann::json{
        {"size", v.size}, {"front", v.front}, {"delta", v.delta}};
}

void from_json(nlohmann::json const& j, UniformGridData& v)
{
    j.at("size").get_to(v.size);
    j.at("front").get_to(v.front);
    j.at("delta").get_to(v.delta);
    v.back = v.front + v.delta * (v.size - 1);
}
}  // namespace celeritas
