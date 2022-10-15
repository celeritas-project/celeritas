//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/Transporter.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Transporter.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
//! Save data to json
inline void to_json(nlohmann::json& j, const TransporterTiming& v)
{
    j = nlohmann::json{{"steps", v.steps},
                       {"total", v.total},
                       {"setup", v.setup},
                       {"actions", v.actions}};
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
} // namespace demo_loop
