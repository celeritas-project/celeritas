//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SenseEvaluator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/VariantUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/surf/VariantSurface.hh"

#include "../CsgTree.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate whether a point is inside a CSG tree node.
 *
 * This is a construction-time helper that combines \c SenseCalculator with \c
 * LogicEvaluator. Its intended use is primarily for testing.
 */
class SenseEvaluator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = SignedSense;
    using VecSurface = std::vector<VariantSurface>;
    //!@}

  public:
    // Construct with a CSG tree, surfaces, and a point
    inline SenseEvaluator(CsgTree const& tree,
                          VecSurface const& surfaces,
                          Real3 const& pos);

    //! Visit from a node ID
    result_type operator()(NodeId const& n) const;

    //!@{
    //! \name Visit a node directly

    //! Point is always inside
    result_type operator()(True const&) const { return SignedSense::inside; }
    //! Point is always outside
    result_type operator()(False const&) const { return SignedSense::outside; }
    // Evaluate a surface
    result_type operator()(Surface const&) const;
    // Redirect an alias
    result_type operator()(Aliased const&) const;
    // Negate the daughter result
    result_type operator()(Negated const&) const;
    // Visit daughter nodes using short circuit logic
    result_type operator()(Joined const&) const;
    //!@}

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const& surfaces_;
    Real3 pos_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a CSG tree and the position to test.
 */
SenseEvaluator::SenseEvaluator(CsgTree const& tree,
                               VecSurface const& surfaces,
                               Real3 const& pos)
    : visit_node_{tree}, surfaces_{surfaces}, pos_{pos}
{
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
