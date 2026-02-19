//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string_view>
#include <vector>

#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Stream a logic token
void logic_to_stream(std::ostream& os, logic_int val);

// Convert a logic vector to a string.
std::string logic_to_string(std::vector<logic_int> const& logic);

// Build a logic definition from a string
std::vector<logic_int> string_to_logic(std::string_view s);

// Get a vector of logic indicating "nowhere"
std::vector<logic_int> make_nowhere_expr(LogicNotation notation);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
