//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/LogicUtils.cc
//---------------------------------------------------------------------------//
#include "LogicUtils.hh"

#include <vector>

#include "corecel/Assert.hh"

#include "../OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Return true if the token is an operand (surface ID or true).
 */
inline bool is_operand_token(logic_int token)
{
    return !logic::is_operator_token(token) || token == logic::ltrue;
}

//---------------------------------------------------------------------------//
/*!
 * Return the precedence of the given operator.
 */
inline int precedence(logic_int token)
{
    switch (token)
    {
        case logic::lor:
            return 1;
        case logic::land:
            return 2;
        case logic::lnot:
            return 3;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return true if the operator is right associative.
 */
inline bool is_right_associative(logic_int token)
{
    return token == logic::lnot;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 *
 * This is a helper class for \c convert_to_infix. It provides helper
 * functions for building the infix expression using a stack.
 */
class InfixStack
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

//---------------------------------------------------------------------------//
/*!
 * Convert an infix logic expression to a postfix expression.
 *
 * This is a helper class for \c convert_to_postfix. It provides helper
 * functions for building the postfix expression using a stack.
 */
class PostfixStack
{
  public:
    using size_type = std::vector<logic_int>::size_type;

    void reserve(size_type size)
    {
        postfix_.reserve(size);
        operators_.reserve(size);
    }

    void push_operand(logic_int token)
    {
        CELER_EXPECT(is_operand_token(token));
        postfix_.push_back(token);
    }

    void push_open_paren() { operators_.push_back(logic::lopen); }

    void push_close_paren()
    {
        CELER_EXPECT(!operators_.empty());
        while (operators_.back() != logic::lopen)
        {
            postfix_.push_back(operators_.back());
            operators_.pop_back();
        }
        CELER_EXPECT(!operators_.empty());
        operators_.pop_back();
    }

    void push_binary(logic_int token)
    {
        CELER_EXPECT(token == logic::lor || token == logic::land);
        this->pop_ready(token);
        operators_.push_back(token);
    }

    void push_unary(logic_int token)
    {
        CELER_EXPECT(token == logic::lnot);
        this->pop_ready(token);
        operators_.push_back(token);
    }

    std::vector<logic_int> get_postfix() &&
    {
        while (!operators_.empty())
        {
            CELER_ASSERT(operators_.back() != logic::lopen);
            postfix_.push_back(operators_.back());
            operators_.pop_back();
        }

        CELER_ENSURE(!postfix_.empty());
        return std::move(postfix_);
    }

  private:
    void pop_ready(logic_int token)
    {
        int const prec = precedence(token);
        while (!operators_.empty())
        {
            logic_int const top = operators_.back();
            if (top == logic::lopen)
            {
                break;
            }

            int const top_prec = precedence(top);
            if (top_prec > prec
                || (top_prec == prec && !is_right_associative(token)))
            {
                postfix_.push_back(top);
                operators_.pop_back();
                continue;
            }
            break;
        }
    }

    std::vector<logic_int> postfix_;
    std::vector<logic_int> operators_;
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

    InfixStack infix_expr;

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
 * Convert an infix logic expression to a postfix expression.
 */
std::vector<logic_int> convert_to_postfix(Span<logic_int const> infix)
{
    CELER_EXPECT(!infix.empty());

    PostfixStack postfix;
    postfix.reserve(infix.size());

    bool expect_operand = true;

    for (logic_int token : infix)
    {
        if (is_operand_token(token))
        {
            CELER_EXPECT(expect_operand);
            postfix.push_operand(token);
            expect_operand = false;
            continue;
        }

        switch (token)
        {
            case logic::lopen:
                CELER_ASSERT(expect_operand);
                postfix.push_open_paren();
                break;
            case logic::lclose:
                CELER_ASSERT(!expect_operand);
                postfix.push_close_paren();
                expect_operand = false;
                break;
            case logic::lor:
            case logic::land:
                CELER_ASSERT(!expect_operand);
                postfix.push_binary(token);
                expect_operand = true;
                break;
            case logic::lnot:
                CELER_ASSERT(expect_operand);
                postfix.push_unary(token);
                expect_operand = true;
                break;
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }

    CELER_ENSURE(!expect_operand);

    return std::move(postfix).get_postfix();
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
/*!
 * Convert logic expressions in an OrangeInput to the desired notation.
 */
void convert_logic(OrangeInput& input, LogicNotation to)
{
    CELER_EXPECT(input);
    if (input.logic == to)
    {
        return;
    }
    CELER_ASSERT(input.logic == LogicNotation::postfix
                 || input.logic == LogicNotation::infix);

    auto convert_units = [&](auto&& converter) {
        for (auto& univ : input.universes)
        {
            if (auto* unit = std::get_if<UnitInput>(&univ))
            {
                for (auto& vol : unit->volumes)
                {
                    if (vol.logic.empty())
                    {
                        continue;
                    }
                    vol.logic = converter(make_span(vol.logic));
                }
            }
        }
    };

    switch (to)
    {
        case LogicNotation::postfix:
            CELER_ASSERT(input.logic == LogicNotation::infix);
            convert_units([](Span<logic_int const> logic) {
                return convert_to_postfix(logic);
            });
            break;
        case LogicNotation::infix:
            CELER_ASSERT(input.logic == LogicNotation::postfix);
            convert_units([](Span<logic_int const> logic) {
                return convert_to_infix(logic);
            });
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    input.logic = to;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
