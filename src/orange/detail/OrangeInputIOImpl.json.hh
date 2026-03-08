//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrangeInputIOImpl.json.hh
//! \sa LogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <nlohmann/json.hpp>

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

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
