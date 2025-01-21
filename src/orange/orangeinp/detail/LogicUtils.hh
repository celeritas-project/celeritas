//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
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

//---------------------------------------------------------------------------//
/*!
 * A helper class for keeping track of the operand type of a sub-expression.
 */
struct Operand
{
    logic::OperatorToken expr_type;
    std::vector<logic_int> expr;
};

//---------------------------------------------------------------------------//
/*!
 * Convert a postfix logic expression to an infix expression.
 *
 * The \c InfixEvaluator will short-circuit evaluation of operands based
 * on parenthesis depth. Minimizing that depth in the expression
 * will allow to short-circuit more efficiently.
 */
inline std::vector<logic_int> convert_to_infix(Span<logic_int> postfix)
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
/*!
 * Result of building a logic representation of a node.
 */
struct BuildLogicResult
{
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;

    VecSurface faces;
    VecLogic logic;
};

//---------------------------------------------------------------------------//
/*!
 * Sort the faces of a volume and remap the logic expression.
 */
inline BuildLogicResult::VecSurface remap_faces(BuildLogicResult::VecLogic& lgc)
{
    // Construct sorted vector of faces
    BuildLogicResult::VecSurface faces;
    for (auto const& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            faces.push_back(LocalSurfaceId{v});
        }
    }

    // Sort and uniquify the vector
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());

    // Remap logic
    for (auto& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            auto iter
                = find_sorted(faces.begin(), faces.end(), LocalSurfaceId{v});
            CELER_ASSUME(iter != faces.end());
            v = iter - faces.begin();
        }
    }
    return faces;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a logic representation of a node.
 *
 * The result is a pair of vectors: the sorted surface IDs comprising the faces
 * of this volume, and the logical representation using \em face IDs, i.e. with
 * the surfaces remapped to the index of the surface in the face vector.
 *
 * The function is templated on a policy class that determines the logic
 * representation. The policy class must have an operator() that takes a
 * NodeId.
 *
 * The per-node local surfaces (faces) are sorted in ascending order of ID, not
 * of access, since they're always evaluated sequentially rather than as part
 * of the logic evaluation itself.
 */
template<class BuildLogicPolicy>
inline BuildLogicResult build_logic(BuildLogicPolicy&& policy, NodeId n)
{
    static_assert(std::is_invocable_v<BuildLogicPolicy, NodeId>);
    static_assert(std::is_rvalue_reference_v<BuildLogicPolicy&&>,
                  "Will move from policy: rvalue ref expected");

    // Construct logic vector as local surface IDs
    auto& lgc = policy.logic();
    CELER_EXPECT(lgc.empty());
    policy(n);

    return {remap_faces(lgc), std::move(lgc)};
}

//---------------------------------------------------------------------------//
/*!
 * Base class for logic builder policies following CRTP pattern.
 *
 * The call operator for Negated and Joined are not implemented in the base
 * policy and must be provided by the derived class.
 */
template<class BuilderPolicy>
class BaseBuildLogicPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

    static_assert(std::is_same_v<LocalSurfaceId::size_type, logic_int>,
                  "unsupported: add enum logic conversion for different-sized "
                  "face and surface ints");

  public:
    // Construct without mapping
    explicit inline BaseBuildLogicPolicy(CsgTree const& tree);
    // Construct with optional mapping
    inline BaseBuildLogicPolicy(CsgTree const& tree, VecSurface const& vs);

    //! Build from a node ID
    inline void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    // Append 'true'
    inline void operator()(True const&);
    // False is never explicitly part of the node tree
    inline void operator()(False const&);
    // Append a surface ID
    inline void operator()(Surface const&);
    // Aliased nodes should never be reachable explicitly
    inline void operator()(Aliased const&);
    //!@}

    //! Access the logic expression
    VecLogic& logic() { return logic_; }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_{nullptr};
    VecLogic logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct without mapping.
 */
template<class BuilderPolicy>
BaseBuildLogicPolicy<BuilderPolicy>::BaseBuildLogicPolicy(CsgTree const& tree)
    : visit_node_{tree}
{
    static_assert(std::is_base_of_v<BaseBuildLogicPolicy, BuilderPolicy>,
                  "CRTP: template parameter must be derived class");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional mapping.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 */
template<class BuilderPolicy>
BaseBuildLogicPolicy<BuilderPolicy>::BaseBuildLogicPolicy(CsgTree const& tree,
                                                          VecSurface const& vs)
    : visit_node_{tree}, mapping_{&vs}
{
    static_assert(std::is_base_of_v<BaseBuildLogicPolicy, BuilderPolicy>,
                  "CRTP: template parameter must be derived class");
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
template<class BuilderPolicy>
void BaseBuildLogicPolicy<BuilderPolicy>::operator()(NodeId const& n)
{
    visit_node_(static_cast<BuilderPolicy&>(*this), n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
template<class BuilderPolicy>
void BaseBuildLogicPolicy<BuilderPolicy>::operator()(True const&)
{
    logic_.push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
template<class BuilderPolicy>
void BaseBuildLogicPolicy<BuilderPolicy>::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
template<class BuilderPolicy>
void BaseBuildLogicPolicy<BuilderPolicy>::operator()(Surface const& s)
{
    CELER_EXPECT(s.id < logic::lbegin);
    // Get index of original surface or remapped
    logic_int sidx = [this, sid = s.id] {
        if (!mapping_)
        {
            return sid.unchecked_get();
        }
        else
        {
            // Remap by finding position of surface in our mapping
            auto iter = find_sorted(mapping_->begin(), mapping_->end(), sid);
            CELER_ASSERT(iter != mapping_->end());
            return logic_int(iter - mapping_->begin());
        }
    }();

    logic_.push_back(sidx);
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * Aliased node shouldn't be reachable if the tree is fully simplified, but
 * could be reachable for testing purposes.
 */
template<class BuilderPolicy>
void BaseBuildLogicPolicy<BuilderPolicy>::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with postfix operation.
 *
 * This is a policy used as template parameter of the \c build_logic function.
 * The user invokes this class with a node ID (usually representing a cell),
 * and then this class recurses into the daughters using a tree visitor.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "0 1 & 2 & &"}
    all(1, 3, !all(2, 4)) -> {{1, 2, 3, 4}, "0 2 & 1 3 & ~ &"}
 * \endverbatim
 */
class PostfixBuildLogicPolicy
    : public BaseBuildLogicPolicy<PostfixBuildLogicPolicy>
{
  public:
    //!@{
    //! \name Type aliases
    using BaseBuildLogicPolicy::VecLogic;
    using BaseBuildLogicPolicy::VecSurface;
    //!@}

  public:
    using BaseBuildLogicPolicy::BaseBuildLogicPolicy;

    //!@{
    //! \name Visit a node directly
    using BaseBuildLogicPolicy::operator();

    //! Visit a negated node and append 'not'.
    void operator()(Negated const& n)
    {
        (*this)(n.node);
        logic().push_back(logic::lnot);
    }

    //! Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
    {
        CELER_EXPECT(n.nodes.size() > 1);

        // Visit first node, then add conjunction for subsequent nodes
        auto iter = n.nodes.begin();
        (*this)(*iter++);

        while (iter != n.nodes.end())
        {
            (*this)(*iter++);
            logic().push_back(n.op);
        }
    }
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with infix operation.
 *
 * This is a policy used as template parameter of \c build_logic.
 * The user invokes this class with a node ID (usually representing a cell),
 * and then this class recurses into the daughters using a tree visitor.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "(0 & 1 & 2)"}
    all(1, 3, any(~(2), ~(4))) -> {{1, 2, 3, 4}, "(0 & 2 & (~1 | ~3))"}
 * \endverbatim
 */
class InfixBuildLogicPolicy : public BaseBuildLogicPolicy<InfixBuildLogicPolicy>
{
  public:
    //!@{
    //! \name Type aliases
    using BaseBuildLogicPolicy::VecLogic;
    using BaseBuildLogicPolicy::VecSurface;
    //!@}

  public:
    using BaseBuildLogicPolicy::BaseBuildLogicPolicy;

    //!@{
    //! \name Visit a node directly
    using BaseBuildLogicPolicy::operator();

    //! Append 'not' and visit a negated node.
    void operator()(Negated const& n)
    {
        this->logic().push_back(logic::lnot);
        (*this)(n.node);
    }

    //! Visit daughter nodes and append the conjunction.
    void operator()(Joined const& n)
    {
        CELER_EXPECT(n.nodes.size() > 1);
        auto& logic = this->logic();
        logic.push_back(logic::lopen);
        // Visit first node, then add conjunction for subsequent nodes
        auto iter = n.nodes.begin();
        (*this)(*iter++);

        while (iter != n.nodes.end())
        {
            logic.push_back(n.op);
            (*this)(*iter++);
        }
        logic.push_back(logic::lclose);
    }
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
