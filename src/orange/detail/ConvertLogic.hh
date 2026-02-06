//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Span.hh"
#include "orange/OrangeInput.hh"

#include "../OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Convert a postfix logic expression to an infix expression.
std::vector<logic_int> convert_to_infix(Span<logic_int const> postfix);

// Convert an infix logic expression to a postfix expression.
std::vector<logic_int> convert_to_postfix(Span<logic_int const> infix);

// Convert logic expressions in an OrangeInput to the desired notation
void convert_logic(OrangeInput& input, LogicNotation to);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
