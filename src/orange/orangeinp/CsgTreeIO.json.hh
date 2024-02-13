//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "CsgTree.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
