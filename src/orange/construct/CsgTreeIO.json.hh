//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/CsgTreeIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "CsgTree.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

void to_json(nlohmann::json& j, CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace celeritas
