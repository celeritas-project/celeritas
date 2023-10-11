//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/OrangeInputIOImpl.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "orange/OrangeTypes.hh"
#include "orange/surf/VariantSurface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

// Read a vector of surfaces
std::vector<VariantSurface> read_surfaces(nlohmann::json const& j);

// Build a logic definition from a C string.
std::vector<logic_int> string_to_logic(std::string const& s);

// Construct a transform from a translation.
VariantTransform make_transform(Real3 const& translation);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
