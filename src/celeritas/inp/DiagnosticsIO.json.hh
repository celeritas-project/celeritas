//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/DiagnosticsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Diagnostics.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, ExportFiles const&);
void from_json(nlohmann::json const& j, ExportFiles&);

void to_json(nlohmann::json& j, SlotDiagnostic const&);
void from_json(nlohmann::json const& j, SlotDiagnostic&);

void to_json(nlohmann::json& j, Timers const&);
void from_json(nlohmann::json const& j, Timers&);

void to_json(nlohmann::json& j, Counters const&);
void from_json(nlohmann::json const& j, Counters&);

void to_json(nlohmann::json& j, McTruth const&);
void from_json(nlohmann::json const& j, McTruth&);

void to_json(nlohmann::json& j, StepDiagnostic const&);
void from_json(nlohmann::json const& j, StepDiagnostic&);

void to_json(nlohmann::json& j, Diagnostics const&);
void from_json(nlohmann::json const& j, Diagnostics&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
