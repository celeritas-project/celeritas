//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ScoringIO.json.cc
//---------------------------------------------------------------------------//
#include "ScoringIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/LabelIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON
//! \todo Add IO for \c GeandSd, \c OpticalDetector

void to_json(nlohmann::json& j, SimpleCalo const& v)
{
    j = {CELER_JSON_PAIR(v, volumes)};
}

void from_json(nlohmann::json const& j, SimpleCalo& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, volumes);
}

void to_json(nlohmann::json& j, Scoring const& v)
{
    j = {CELER_JSON_PAIR_OPTIONAL(v, simple_calo)};
}

void from_json(nlohmann::json const& j, Scoring& v)
{
    CELER_JSON_LOAD_OPTIONAL(j, v, simple_calo);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
