//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Join.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a logic token to a string.
 */
inline void logic_to_stream(std::ostream& os, logic_int val)
{
    if (logic::is_operator_token(val))
    {
        os << to_char(static_cast<logic::OperatorToken>(val));
    }
    else
    {
        // Just a face ID
        os << val;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert a logic vector to a string.
 */
inline std::string logic_to_string(std::vector<logic_int> const& logic)
{
    return to_string(celeritas::join_stream(
        logic.begin(), logic.end(), ' ', logic_to_stream));
}

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 */
std::vector<logic_int> convert_to_infix(Span<logic_int const> postfix);

//---------------------------------------------------------------------------//
/*!
 * Build a logic definition from a C string.
 */
std::vector<logic_int> string_to_logic(std::string const& s);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
