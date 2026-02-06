//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ProblemIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Problem.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, OpticalProblem const&);
void from_json(nlohmann::json const& j, OpticalProblem&);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
