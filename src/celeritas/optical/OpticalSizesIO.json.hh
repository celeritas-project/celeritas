//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalSizesIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"

#include "OpticalSizes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, OpticalSizes const& v)
{
    j = {
        CELER_JSON_PAIR(v, primaries),
        CELER_JSON_PAIR(v, tracks),
        CELER_JSON_PAIR(v, generators),
        CELER_JSON_PAIR(v, streams),
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
