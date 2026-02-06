//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicIO.cc
//---------------------------------------------------------------------------//
#include "LogicIO.hh"

#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/io/Join.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a logic token to a string.
 */
void logic_to_stream(std::ostream& os, logic_int val)
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
std::string logic_to_string(std::vector<logic_int> const& logic)
{
    return to_string(celeritas::join_stream(
        logic.begin(), logic.end(), ' ', logic_to_stream));
}

//---------------------------------------------------------------------------//
/*!
 * Build a logic definition from a string.
 *
 * A valid string satisfies the regex "[0-9~!| ()]+", and must have balanced
 * parentheses, but the result may not be a valid logic expression and its
 * interpretation depends on the logic notation.
 *
 * \par Example
 * \code

     string_to_logic("4 ~ 5 & 6 &");

   \endcode
 */
std::vector<logic_int> string_to_logic(std::string_view s)
{
    std::vector<logic_int> result;

    logic_int surf_id{};
    bool reading_surf{false};
    int parens_depth{0};
    for (char v : s)
    {
        if (v >= '0' && v <= '9')
        {
            // Parse a surface number. 'Push' this digit onto the surface ID by
            // multiplying the existing ID by 10.
            if (!reading_surf)
            {
                surf_id = 0;
                reading_surf = true;
            }
            surf_id = 10 * surf_id + (v - '0');
            continue;
        }
        else if (reading_surf)
        {
            // Next char is end of word or end of string
            result.push_back(surf_id);
            reading_surf = false;
        }

        // Parse a logic token
        // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
        switch (v)
        {
            case '(':
                result.push_back(logic::lopen);
                ++parens_depth;
                continue;
            case ')':
                CELER_VALIDATE(parens_depth > 0,
                               << "unmatched ')' in logic string");
                --parens_depth;
                result.push_back(logic::lclose);
                continue;
                // clang-format off
            case '|': result.push_back(logic::lor);   continue;
            case '&': result.push_back(logic::land);  continue;
            case '~': result.push_back(logic::lnot);  continue;
            case '*': result.push_back(logic::ltrue); continue;
                // clang-format on
        }
        CELER_VALIDATE(v == ' ',
                       << "unexpected token '" << v
                       << "' while parsing logic string");
    }
    if (reading_surf)
    {
        result.push_back(surf_id);
    }

    CELER_VALIDATE(parens_depth == 0, << "unmatched '(' in logic string");

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a vector of logic indicating "nowhere".
 */
std::vector<logic_int> make_nowhere_expr(LogicNotation notation)
{
    CELER_EXPECT(notation != LogicNotation::size_);

    switch (notation)
    {
        case LogicNotation::postfix:
            return {logic::ltrue, logic::lnot};
        case LogicNotation::infix:
            return {logic::lnot, logic::ltrue};
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
