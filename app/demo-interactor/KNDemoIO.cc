//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoIO.cc
//---------------------------------------------------------------------------//
#include "KNDemoIO.hh"

#include <nlohmann/json.hpp>

namespace demo_interactor
{
//---------------------------------------------------------------------------//
//@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, const CudaGridParams& v)
{
    j = nlohmann::json{{"block_size", v.block_size},
                       {"grid_size", v.grid_size}};
}

void from_json(const nlohmann::json& j, CudaGridParams& v)
{
    j.at("block_size").get_to(v.block_size);
    j.at("grid_size").get_to(v.grid_size);
}

void to_json(nlohmann::json& j, const KNDemoRunArgs& v)
{
    j = nlohmann::json{{"energy", v.energy},
                       {"seed", v.seed},
                       {"num_tracks", v.num_tracks},
                       {"max_steps", v.max_steps}};
}

void from_json(const nlohmann::json& j, KNDemoRunArgs& v)
{
    j.at("energy").get_to(v.energy);
    j.at("seed").get_to(v.seed);
    j.at("num_tracks").get_to(v.num_tracks);
    j.at("max_steps").get_to(v.max_steps);
}

void to_json(nlohmann::json& j, const KNDemoResult& v)
{
    j = nlohmann::json{{"time", v.time},
                       {"edep", v.edep},
                       {"alive", v.alive},
                       {"total_time", v.total_time}};
}

void from_json(const nlohmann::json& j, KNDemoResult& v)
{
    j.at("time").get_to(v.time);
    j.at("edep").get_to(v.edep);
    j.at("alive").get_to(v.alive);
    j.at("total_time").get_to(v.total_time);
}
//@}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
