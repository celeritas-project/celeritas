//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHTraversal.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse BIH tree using a depth-first search.
 *
 * \todo move to top-level orange directory out of detail namespace
 */
class BIHTraversalHelper
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHTraversalHelper(BIHTree const& tree, Storage const& storage);

    // Determine if a node is inner, i.e., not a leaf
    inline CELER_FUNCTION bool is_inner(BIHNodeId id) const;

    // Get an inner node for a given BIHNodeId
    inline CELER_FUNCTION BIHInnerNode const&
    get_inner_node(BIHNodeId id) const;

    // Get a leaf node for a given BIHNodeId
    inline CELER_FUNCTION BIHLeafNode const& get_leaf_node(BIHNodeId id) const;

  private:
    //// DATA ////
    BIHTree const& tree_;
    Storage const& storage_;
    size_type leaf_offset_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHTraversalHelper::BIHTraversalHelper(
    BIHTree const& tree, BIHTraversalHelper::Storage const& storage)
    : tree_(tree), storage_(storage), leaf_offset_(tree.inner_nodes.size())
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 *  Determine if a node is inner, i.e., not a leaf.
 */
CELER_FUNCTION
bool BIHTraversalHelper::is_inner(BIHNodeId id) const
{
    return id.unchecked_get() < leaf_offset_;
}

//---------------------------------------------------------------------------//
/*!
 *  Get an inner node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHInnerNode const& BIHTraversalHelper::get_inner_node(BIHNodeId id) const
{
    CELER_EXPECT(this->is_inner(id));
    return storage_.inner_nodes[tree_.inner_nodes[id.unchecked_get()]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get a leaf node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHLeafNode const& BIHTraversalHelper::get_leaf_node(BIHNodeId id) const
{
    CELER_EXPECT(!this->is_inner(id));
    return storage_
        .leaf_nodes[tree_.leaf_nodes[id.unchecked_get() - leaf_offset_]];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
