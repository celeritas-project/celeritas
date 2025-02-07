//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/CoreSizes.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "corecel/Types.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Save state size/capacity counters.
 *
 * These should be *integrated* across streams, *per process*.
 */
struct CoreSizes
{
    size_type initializers{};
    size_type tracks{};
    size_type secondaries{};
    size_type events{};

    size_type streams{};
    size_type processes{};
};

//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, CoreSizes const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, initializers),
        CELER_JSON_PAIR(inp, tracks),
        CELER_JSON_PAIR(inp, secondaries),
        CELER_JSON_PAIR(inp, events),
        CELER_JSON_PAIR(inp, streams),
        CELER_JSON_PAIR(inp, processes),
    };
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
