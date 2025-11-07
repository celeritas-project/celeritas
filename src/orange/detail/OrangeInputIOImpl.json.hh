//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrangeInputIOImpl.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "orange/OrangeTypes.hh"
#include "orange/surf/VariantSurface.hh"
#include "orange/transform/VariantTransform.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Read a transform from a JSON object
VariantTransform import_transform(nlohmann::json const& src);

// Write a transform to arrays suitable for JSON export.
nlohmann::json export_transform(VariantTransform const& t);

// Read surface data from a JSON object
std::vector<VariantSurface> import_zipped_surfaces(nlohmann::json const& j);

// Write surface data to a JSON object
nlohmann::json export_zipped_surfaces(std::vector<VariantSurface> const& s);

// Build a logic definition from a C string.
std::vector<logic_int> string_to_logic(std::string const& s);

// Convert a logic vector to a string
std::string logic_to_string(std::vector<logic_int> const&);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
