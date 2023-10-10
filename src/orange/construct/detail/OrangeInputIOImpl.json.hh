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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Convert a surface type string to an enum for I/O.
SurfaceType to_surface_type(std::string const& s);

// Build a logic definition from a C string.
std::vector<logic_int> parse_logic(char const*);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
