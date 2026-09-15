//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/inp/GridIO.json.cc
//---------------------------------------------------------------------------//
#include "GridIO.json.hh"

#include "corecel/grid/GridTypesIO.json.hh"
#include "corecel/io/JsonUtils.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, Interpolation const& v)
{
    j = {
        CELER_JSON_PAIR(v, type),
        CELER_JSON_PAIR(v, order),
        CELER_JSON_PAIR(v, bc),
    };
}

void from_json(nlohmann::json const& j, Interpolation& v)
{
    CELER_JSON_LOAD_OPTION(j, v, type);
    CELER_JSON_LOAD_OPTION(j, v, order);
    CELER_JSON_LOAD_OPTION(j, v, bc);
}

//!@}
//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
