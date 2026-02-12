//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/ConvertLogic.cc
//---------------------------------------------------------------------------//
#include "ConvertLogic.hh"

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTreeUtils.hh"
#include "orange/orangeinp/detail/BuildLogic.hh"

#include "../OrangeTypes.hh"

using VecLogic = std::vector<celeritas::logic_int>;

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
orangeinp::CsgTree build_tree_from_postfix(VecLogic const& postfix)
{
    using orangeinp::CsgTree;
    using orangeinp::Joined;
    using orangeinp::Negated;
    using orangeinp::Node;
    using orangeinp::NodeId;
    using orangeinp::True;

    CELER_EXPECT(!postfix.empty());

    CsgTree tree;
    std::vector<NodeId> stack;
    stack.reserve(postfix.size());

    for (logic_int token : postfix)
    {
        if (!logic::is_operator_token(token))
        {
            auto [node_id, inserted] = tree.insert(LocalSurfaceId{token});
            stack.push_back(node_id);
            continue;
        }

        switch (token)
        {
            case logic::ltrue: {
                auto [node_id, inserted] = tree.insert(Node{True{}});
                stack.push_back(node_id);
                break;
            }
            case logic::lnot: {
                CELER_ASSERT(!stack.empty());
                auto child = stack.back();
                stack.pop_back();
                auto [node_id, inserted] = tree.insert(Node{Negated{child}});
                stack.push_back(node_id);
                break;
            }
            case logic::lor:
                [[fallthrough]];
            case logic::land: {
                CELER_ASSERT(stack.size() >= 2);
                auto right = stack.back();
                stack.pop_back();
                auto left = stack.back();
                stack.pop_back();
                auto [node_id, inserted] = tree.insert(
                    Node{Joined{logic::OperatorToken{token}, {left, right}}});
                stack.push_back(node_id);
                break;
            }
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }

    CELER_ASSERT(stack.size() == 1);
    auto root = stack.back();
    tree.insert_volume(root);
    return tree;
}

VecLogic simplify_negated_joins_postfix(VecLogic const& postfix)
{
    CELER_EXPECT(!postfix.empty());

    using namespace orangeinp;

    // Construct a CSG tree from the input and simplify it
    CsgTree tree
        = transform_negated_joins(build_tree_from_postfix(postfix)).first;
    CELER_ASSERT(tree.volumes().size() == 1);
    NodeId root = tree.volumes().front();

    // Convert simplified tree to postfix
    orangeinp::detail::PostfixBuildLogicPolicy const make_builder{tree};
    VecLogic logic;
    make_builder(logic)(root);
    return logic;
}

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
constexpr bool is_right_associative(logic_int token)
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
        VecLogic expr;
    };

    //! Push a binary operator.
    void push_binary(logic_int op)
    {
        CELER_EXPECT(infix_.size() > 1);
        auto& op_2 = infix_.back();
        auto& op_1 = *(infix_.end() - 2);
        VecLogic new_expr;
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
        VecLogic new_expr;
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
    VecLogic get_infix() &&
    {
        CELER_EXPECT(infix_.size() == 1);
        return std::move(infix_.front().expr);
    }

  private:
    //! Accumulate operands into a new expression.
    void add_sub_expr(VecLogic& acc, VecLogic const& expr, bool parentheses)
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
    using size_type = VecLogic::size_type;

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

    VecLogic get_postfix() &&
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

    VecLogic postfix_;
    VecLogic operators_;
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 *
 * The \c InfixEvaluator will short-circuit evaluation of operands based
 * on parenthesis depth. Minimizing that depth in the expression
 * will allow to short-circuit more efficiently.
 */
VecLogic convert_to_infix(VecLogic const& postfix)
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
VecLogic convert_to_postfix(VecLogic const& infix)
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
 * Convert logic expressions in an OrangeInput to the desired notation.
 */
void convert_logic(OrangeInput& input, LogicNotation target)
{
    CELER_EXPECT(input);
    CELER_ASSERT(input.logic != LogicNotation::size_);
    CELER_ASSERT(target != LogicNotation::size_);

    if (input.logic == target)
    {
        // No conversion necessary
        return;
    }

    ScopedProfiling profile_this{"orange-logic-convert"};

    std::function<VecLogic(VecLogic const&)> convert;

    switch (target)
    {
        case LogicNotation::postfix:
            convert = convert_to_postfix;
            break;
        case LogicNotation::infix:
            convert = [](VecLogic const& postfix_lgc) {
                auto simplified = simplify_negated_joins_postfix(postfix_lgc);
                return convert_to_infix(simplified);
            };
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

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
                vol.logic = convert(vol.logic);
            }
        }
    }
    input.logic = target;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
