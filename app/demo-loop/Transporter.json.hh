//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Transporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Save data to json
inline void to_json(nlohmann::json& j, const TransporterTiming& v)
{
    j = nlohmann::json{{"steps", v.steps},
                       {"total", v.total},
                       {"setup", v.setup},
                       {"initialize_tracks", v.initialize_tracks},
                       {"pre_step", v.pre_step},
                       {"along_step", v.along_step},
                       {"discrete_select", v.discrete_select},
                       {"cross_boundary", v.launch_models},
                       {"launch_models", v.launch_models},
                       {"extend_from_secondaries", v.extend_from_secondaries}};
}

inline void to_json(nlohmann::json& j, const TransporterResult& v)
{
    j = nlohmann::json{{"initializers", v.initializers},
                       {"active", v.active},
                       {"alive", v.alive},
                       {"edep", v.edep},
                       {"process", v.process},
                       {"steps", v.steps},
                       {"time", v.time}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
