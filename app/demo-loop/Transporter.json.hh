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
inline void to_json(nlohmann::json& j, const TransporterResult& v)
{
    j = nlohmann::json{{"time", v.time},
                       {"alive", v.alive},
                       {"edep", v.edep},
                       {"process", v.process},
                       {"steps", v.steps},
                       {"total_time", v.total_time}};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
