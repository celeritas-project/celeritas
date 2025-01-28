//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.cc
//---------------------------------------------------------------------------//
#include "LogicUtils.hh"

#include <vector>

#include "corecel/Assert.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 *
 * This is a helper class for \c convert_to_infix. It provides helper
 * functions for building the infix expression using a stack.
 */
class ExprStack
{
  public:
    //! A helper class for keeping track of the operand type of a
    //! sub-expression.
    struct Operand
    {
        logic::OperatorToken expr_type;
        std::vector<logic_int> expr;
    };

    //! Push a binary operator.
    void push_binary(logic_int op)
    {
        CELER_EXPECT(infix_.size() > 1);
        auto& op_2 = infix_.back();
        auto& op_1 = *(infix_.end() - 2);
        std::vector<logic_int> new_expr;
        constexpr int max_extra_tokens = 5;
        new_expr.reserve(max_extra_tokens + op_1.expr.size()
                         + op_2.expr.size());
        auto opposite = op == logic::lor ? logic::land : logic::lor;
        this->add_sub_expr(new_expr, op_1.expr, opposite == op_1.expr_type);
        new_expr.push_back(op);
        this->add_sub_expr(new_expr, op_2.expr, opposite == op_2.expr_type);
        infix_.pop_back();
        infix_.pop_back();
        infix_.push_back({logic::OperatorToken{op}, new_expr});
    }

    //! Push an operand.
    void push_unary(logic_int op)
    {
        CELER_EXPECT(!infix_.empty());
        auto&& [expr_type, expr] = infix_.back();
        std::vector<logic_int> new_expr;
        constexpr int max_extra_tokens = 3;
        new_expr.reserve(max_extra_tokens + expr.size());

        new_expr.push_back(op);
        this->add_sub_expr(new_expr, expr, expr_type < logic::lnot);
        infix_.pop_back();
        infix_.push_back({logic::OperatorToken{op}, new_expr});
    }

    //! Push a primitive (surface).
    void push_primitive(logic_int elem)
    {
        // hijack ltrue as the token to represent a primitive
        infix_.push_back({logic::ltrue, {elem}});
    }

    //! Get the infix expression.
    std::vector<logic_int> get_infix() &&
    {
        CELER_EXPECT(infix_.size() == 1);
        return std::move(infix_.front().expr);
    }

  private:
    //! Accumulate operands into a new expression.
    void add_sub_expr(std::vector<logic_int>& acc,
                      std::vector<logic_int> const& expr,
                      bool parentheses)
    {
        if (parentheses)
        {
            acc.push_back(logic::lopen);
        }
        acc.insert(acc.end(), expr.begin(), expr.end());
        if (parentheses)
        {
            acc.push_back(logic::lclose);
        }
    }

    //! The infix expression; used as a stack during conversion.
    std::vector<Operand> infix_;
};
}  // namespace

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

    ExprStack infix_expr;

    // Process each token
    for (auto lgc : postfix)
    {
        if (logic::is_operator_token(lgc))
        {
            switch (lgc)
            {
                case logic::ltrue:
                    infix_expr.push_primitive(lgc);
                    break;
                case logic::lor:
                    [[fallthrough]];
                case logic::land: {
                    infix_expr.push_binary(lgc);
                    break;
                }
                case logic::lnot: {
                    infix_expr.push_unary(lgc);
                    break;
                }
                default:
                    CELER_ASSERT_UNREACHABLE();
            }
        }
        else
        {
            infix_expr.push_primitive(lgc);
        }
    }
    return std::move(infix_expr).get_infix();
}

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
}  // namespace detail
}  // namespace celeritas