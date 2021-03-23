//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoIO.cc
//---------------------------------------------------------------------------//
#include "LDemoIO.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
void to_json(nlohmann::json& j, const LDemoArgs& v)
{
    j = nlohmann::json{{"geometry_filename", v.geometry_filename},
                       {"physics_filename", v.physics_filename},
                       {"energy", v.energy},
                       {"seed", v.seed},
                       {"num_tracks", v.num_tracks},
                       {"max_steps", v.max_steps}};
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    j.at("energy").get_to(v.energy);
    j.at("seed").get_to(v.seed);
    j.at("num_tracks").get_to(v.num_tracks);
    j.at("max_steps").get_to(v.max_steps);
}

void to_json(nlohmann::json& j, const LDemoResult& v)
{
    j = nlohmann::json{{"time", v.time},
                       {"alive", v.alive},
                       {"edep", v.edep},
                       {"total_time", v.total_time}};
}
//!@}

//---------------------------------------------------------------------------//
} // namespace demo_loop
