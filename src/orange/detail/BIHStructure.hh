//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHStructure.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <variant>
#include <vector>

#include "corecel/Types.hh"

#include "BIHView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * \brief Diagnostic data class that stores the structure of the BIH tree.
 *
 * This structure consists of a vector of inner/outer nodes, sorted by NodeId,
 * and a list of inf_vols (i.e., volumes with infinite bounding boxes). The
 * inner nodes entries store left/right bounding planes and the left/right
 * child NodeIds. This supports reconstruction of the BIH tree structure from
 * JSON output.
 */
class BIHStructure
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = BIHView::Storage;

    struct Inner
    {
        Axis axis;
        std::array<real_type, 2> bounding_plane_pos;
        std::array<BIHNodeId, 2> children;
    };

    struct Leaf
    {
        std::vector<LocalVolumeId> vol_ids;
    };

    using Node = std::variant<Inner, Leaf>;
    using VecNode = std::vector<Node>;
    using VecLocalVols = std::vector<LocalVolumeId>;
    //!@}

  public:
    // Construct from BIH record and storage
    BIHStructure(BIHTreeRecord const&, Storage const&);

    // Get all inner and leaf nodes
    VecNode const& tree() const { return tree_; }

    // Get the infinite volume IDs
    VecLocalVols const& inf_vol_ids() const { return inf_vol_ids_; }

  private:
    VecNode tree_;
    VecLocalVols inf_vol_ids_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
