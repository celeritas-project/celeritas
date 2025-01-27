//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicUtils.cc
//---------------------------------------------------------------------------//
#include "LogicUtils.hh"

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Join.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * A helper class for keeping track of the operand type of a sub-expression.
 */
struct Operand
{
    logic::OperatorToken expr_type;
    std::vector<logic_int> expr;
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Build a logic definition from a C string.
 *
 * A valid string satisfies the regex "[0-9~!| ]+", but the result may
 * not be a valid logic expression. (The volume inserter will ensure that the
 * logic expression at least is consistent for a CSG region definition.)
 *
 * Example:
 * \code

     parse_logic("4 ~ 5 & 6 &");

   \endcode
 */
std::vector<logic_int> string_to_logic(std::string const& s)
{
    std::vector<logic_int> result;

    logic_int surf_id{};
    bool reading_surf{false};
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
                // clang-format off
            case '*': result.push_back(logic::ltrue); continue;
            case '|': result.push_back(logic::lor);   continue;
            case '&': result.push_back(logic::land);  continue;
            case '~': result.push_back(logic::lnot);  continue;
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

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 *
 * The \c InfixEvaluator will short-circuit evaluation of operands based
 * on parenthesis depth. Minimizing that depth in the expression
 * will allow to short-circuit more efficiently.
 */
std::vector<logic_int> convert_to_infix(Span<logic_int const> postfix)
{
    CELER_EXPECT(!postfix.empty());

    std::vector<Operand> infix;
    infix.reserve(postfix.size());

    auto add_sub_expr = [](std::vector<logic_int>& acc,
                           std::vector<logic_int>& expr,
                           bool parentheses) {
        if (parentheses)
        {
            acc.push_back(logic::lopen);
        }
        acc.insert(acc.end(), expr.begin(), expr.end());
        if (parentheses)
        {
            acc.push_back(logic::lclose);
        }
    };

    // Process each token
    for (auto lgc : postfix)
    {
        if (logic::is_operator_token(lgc))
        {
            switch (lgc)
            {
                case logic::ltrue:
                    infix.push_back({logic::ltrue, {lgc}});
                    break;
                case logic::lor:
                    [[fallthrough]];
                case logic::land: {
                    CELER_EXPECT(infix.size() > 1);
                    auto& op_2 = infix.back();
                    auto& op_1 = *(infix.end() - 2);
                    auto opposite = lgc == logic::lor ? logic::land
                                                      : logic::lor;
                    std::vector<logic_int> new_expr;
                    constexpr int max_extra_tokens = 5;
                    new_expr.reserve(max_extra_tokens + op_1.expr.size()
                                     + op_2.expr.size());
                    add_sub_expr(
                        new_expr, op_1.expr, op_1.expr_type == opposite);
                    new_expr.push_back(lgc);
                    add_sub_expr(
                        new_expr, op_2.expr, op_2.expr_type == opposite);

                    infix.pop_back();
                    infix.pop_back();
                    infix.push_back({logic::OperatorToken{lgc}, new_expr});
                    break;
                }
                case logic::lnot: {
                    CELER_EXPECT(!infix.empty());
                    auto&& [expr_type, expr] = infix.back();
                    std::vector<logic_int> new_expr;
                    constexpr int max_extra_tokens = 3;
                    new_expr.reserve(max_extra_tokens + expr.size());

                    new_expr.push_back(lgc);
                    add_sub_expr(new_expr, expr, expr_type < logic::lnot);

                    infix.pop_back();
                    infix.push_back({logic::lnot, new_expr});
                    break;
                }
                default:
                    CELER_ASSERT_UNREACHABLE();
            }
        }
        else
        {
            infix.push_back({logic::ltrue, {lgc}});
        }
    }
    CELER_ENSURE(infix.size() == 1);
    return infix.front().expr;
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas