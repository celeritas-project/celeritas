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
                       {"hepmc3_filename", v.hepmc3_filename},
                       {"seed", v.seed},
                       {"max_num_tracks", v.max_num_tracks},
                       {"max_steps", v.max_steps},
                       {"storage_factor", v.storage_factor},
                       {"use_device", v.use_device}};
}

void from_json(const nlohmann::json& j, LDemoArgs& v)
{
    j.at("geometry_filename").get_to(v.geometry_filename);
    j.at("physics_filename").get_to(v.physics_filename);
    j.at("hepmc3_filename").get_to(v.hepmc3_filename);
    j.at("seed").get_to(v.seed);
    j.at("max_num_tracks").get_to(v.max_num_tracks);
    j.at("max_steps").get_to(v.max_steps);
    j.at("storage_factor").get_to(v.storage_factor);
    j.at("use_device").get_to(v.use_device);
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
