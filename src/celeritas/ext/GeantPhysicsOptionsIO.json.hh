//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
struct GeantPhysicsOptions;
struct GeantMuonPhysicsOptions;

//---------------------------------------------------------------------------//
// Read options from JSON
void from_json(nlohmann::json const&, GeantPhysicsOptions&);
void from_json(nlohmann::json const&, GeantMuonPhysicsOptions&);

// Write options to JSON
void to_json(nlohmann::json&, GeantPhysicsOptions const&);
void to_json(nlohmann::json&, GeantMuonPhysicsOptions const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
