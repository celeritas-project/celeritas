//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHStructureIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "BIHStructure.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//!@{
//! BIHStructure diagnostic output
void to_json(nlohmann::json& j, BIHStructure::Inner const& value);
void to_json(nlohmann::json& j, BIHStructure::Leaf const& value);
void to_json(nlohmann::json& j, BIHStructure::Node const& value);
void to_json(nlohmann::json& j, BIHStructure const& value);
//!@}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
