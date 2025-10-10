//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/OptionsIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Options.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, InlineSingletons const&);
void from_json(nlohmann::json const& j, InlineSingletons&);

void to_json(nlohmann::json& j, Options const&);
void from_json(nlohmann::json const& j, Options&);

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
