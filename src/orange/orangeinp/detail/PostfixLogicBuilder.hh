//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/PostfixLogicBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a postfix logic representation of a node.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 *
 * The result is a pair of vectors: the sorted surface IDs comprising the faces
 * of this volume, and the logical representation using \em face IDs, i.e. with
 * the surfaces remapped to index of the surface in the face vector.
 *
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "0 1 & 2 & &"}
    all(1, 3, !all(2, 4)) -> {{1, 2, 3, 4}, "0 2 & 1 3 & ~ &"}
 * \endverbatim
 */
class PostfixLogicBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    using result_type = std::pair<VecSurface, VecLogic>;
    //!@}

  public:
    //! Construct from a tree
    explicit PostfixLogicBuilder(CsgTree const& tree) : tree_{tree} {}

    // Construct from a tree and surface remapping
    inline PostfixLogicBuilder(CsgTree const& tree, VecSurface const& old_ids);

    // Convert a single node to postfix notation
    result_type operator()(NodeId n) const;

  private:
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Construct from a tree and surface remapping.
 */
PostfixLogicBuilder::PostfixLogicBuilder(CsgTree const& tree,
                                         VecSurface const& old_ids)
    : tree_{tree}, mapping_{&old_ids}
{
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
