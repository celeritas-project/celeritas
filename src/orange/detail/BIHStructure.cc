//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHStructure.cc
//---------------------------------------------------------------------------//
#include "BIHStructure.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from BIH record and storage.
 */
BIHStructure::BIHStructure(BIHTreeRecord const& tree, Storage const& storage)
{
    CELER_EXPECT(tree);

    tree_.reserve(tree.inner_nodes.size() + tree.leaf_nodes.size());

    // Handle the inner nodes
    for (auto i : range(tree.inner_nodes.size()))
    {
        auto const& inner = storage.inner_nodes[tree.inner_nodes[i]];
        auto const& left_edge = inner.edges[BIHInnerNode::Side::left];
        auto const& right_edge = inner.edges[BIHInnerNode::Side::right];
        tree_.emplace_back(Inner{
            inner.axis,
            {left_edge.bounding_plane_pos, right_edge.bounding_plane_pos},
            {left_edge.child, right_edge.child}});
    }

    // Handle the leaf nodes
    for (auto i : range(tree.leaf_nodes.size()))
    {
        auto const& leaf = storage.leaf_nodes[tree.leaf_nodes[i]];
        auto const& vol_ids = storage.local_volume_ids[leaf.vol_ids];

        Leaf out;
        out.vol_ids.assign(vol_ids.begin(), vol_ids.end());
        tree_.emplace_back(std::move(out));
    }

    auto inf_vol_ids = storage.local_volume_ids[tree.inf_vol_ids];
    inf_vol_ids_.assign(inf_vol_ids.begin(), inf_vol_ids.end());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
