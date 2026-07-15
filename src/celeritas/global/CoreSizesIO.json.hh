//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreSizesIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"

#include "CoreSizes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void to_json(nlohmann::json& j, CoreSizes const& v)
{
    j = {
        CELER_JSON_PAIR(v, primaries),
        CELER_JSON_PAIR(v, tracks),
        CELER_JSON_PAIR(v, initializers),
        CELER_JSON_PAIR(v, secondaries),
        CELER_JSON_PAIR(v, events),
        CELER_JSON_PAIR(v, streams),
        CELER_JSON_PAIR(v, processes),
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
