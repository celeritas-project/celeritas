//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalSizes.json.hh
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
struct OpticalSizes
{
    size_type generators{};
    size_type initializers{};
    size_type tracks{};
    size_type streams{};
};

//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, OpticalSizes const& inp)
{
    j = {
        CELER_JSON_PAIR(inp, generators),
        CELER_JSON_PAIR(inp, initializers),
        CELER_JSON_PAIR(inp, tracks),
        CELER_JSON_PAIR(inp, streams),
    };
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
