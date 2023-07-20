//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include <iostream>
#include <limits>

#include "BoundingBoxUtils.hh"

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sort and uniquify a vector.
 */
void sort_and_uniquify(std::vector<celeritas::real_type>& vec)
{
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}
}  // namespace

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage for LocalVolumeIds.
 */
BIHBuilder::BIHBuilder(VecBBox bboxes,
                       BIHBuilder::LVIStorage* lvi_storage,
                       BIHBuilder::NodeStorage* node_storage)
    : bboxes_(bboxes), lvi_storage_(lvi_storage), node_storage_(node_storage)
{
    CELER_EXPECT(!bboxes.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Create BIH Nodes.
 */
BIHParams BIHBuilder::operator()() const
{
    // Create a vector of indices, excluding index 0 (i.e., the exterior)
    VecIndices indices;
    VecIndices inf_volids;

    for (auto i : range(bboxes_.size()))
    {
        LocalVolumeId id(i);

        if (!is_infinite(bboxes_[i]))
        {
            indices.push_back(id);
        }
        else
        {
            inf_volids.push_back(id);
        }
    }

    VecNodes nodes;
    this->construct_tree(indices, nodes);

    BIHParams params;
    params.nodes
        = make_builder(node_storage_).insert_back(nodes.begin(), nodes.end());
    params.inf_volids = make_builder(lvi_storage_)
                            .insert_back(inf_volids.begin(), inf_volids.end());

    return params;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct BIH nodes for a vector of bbox indices.
 */
void BIHBuilder::construct_tree(VecIndices const& indices, VecNodes& nodes) const
{
    nodes.push_back(BIHNode());
    auto current_index = nodes.size() - 1;

    if (indices.size() > 1)
    {
        auto centers = this->centers(indices);
        auto p = this->find_partition(indices, centers);

        if (p)
        {
            auto ax = to_int(p.axis);

            auto [left_indices, right_indices]
                = partition_bboxes(indices, centers, p);

            CELER_EXPECT(!left_indices.empty() && !right_indices.empty());

            nodes[current_index].partitions
                = {static_cast<BIHNode::partition_location_type>(
                       this->meta_bbox(left_indices).upper()[ax]),
                   static_cast<BIHNode::partition_location_type>(
                       this->meta_bbox(right_indices).lower()[ax])};

            // Recursively construct the left and right branches
            nodes[current_index].children[BIHNode::Edge::left]
                = BIHNodeId(nodes.size());
            this->construct_tree(left_indices, nodes);
            nodes[current_index].children[BIHNode::Edge::right]
                = BIHNodeId(nodes.size());
            this->construct_tree(right_indices, nodes);
        }
        else
        {
            this->make_leaf(nodes[current_index], indices);
        }
    }
    else
    {
        this->make_leaf(nodes[current_index], indices);
    }

    CELER_EXPECT(nodes[current_index]);
}

//---------------------------------------------------------------------------//
/*!
 * Find a suitable partition for the given bounding boxes.
 *
 * If no suitable partition is found an empty Partition object is returned.
 */
BIHBuilder::Partition BIHBuilder::find_partition(VecIndices const& indices,
                                                 VecReal3 const& centers) const
{
    CELER_EXPECT(indices.size() == centers.size());

    auto mb = this->meta_bbox(indices);
    VecAxes sorted_axes = this->sort_axes(mb);
    auto axes_centers = this->axes_centers(centers);

    Partition partition;

    for (Axis axis : sorted_axes)
    {
        auto ax = to_int(axis);

        if (axes_centers[ax].size() < 2)
        {
            continue;
        }
        else
        {
            partition.axis = axis;
            auto size = axes_centers[ax].size();
            partition.location
                = (axes_centers[ax][size / 2 - 1] + axes_centers[ax][size / 2])
                  / 2;
            break;
        }
    }
    return partition;
}

//---------------------------------------------------------------------------//
/*!
 * Divide bboxes into left and right branches based on a partition.
 */
BIHBuilder::PairVecIndices
BIHBuilder::partition_bboxes(VecIndices const& indices,
                             VecReal3 const& centers,
                             Partition const& p) const
{
    CELER_EXPECT(!indices.empty());
    CELER_EXPECT(!centers.empty());
    CELER_EXPECT(indices.size() == centers.size());

    VecIndices left;
    VecIndices right;

    for (auto i : range(indices.size()))
    {
        if (centers[i][to_int(p.axis)] < p.location)
        {
            left.push_back(indices[i]);
        }
        else
        {
            right.push_back(indices[i]);
        }
    }

    return std::make_pair(left, right);
}

//---------------------------------------------------------------------------//
/*!
 * Add leaf volume ids to a given node.
 */
void BIHBuilder::make_leaf(BIHNode& node, VecIndices const& indices) const
{
    CELER_EXPECT(!node);
    CELER_EXPECT(!indices.empty());

    node.vol_ids
        = make_builder(lvi_storage_).insert_back(indices.begin(), indices.end());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the centers of each bounding box.
 */
BIHBuilder::VecReal3 BIHBuilder::centers(VecIndices const& indices) const
{
    CELER_EXPECT(!indices.empty());
    VecReal3 centers(indices.size());
    for (auto i : range(indices.size()))
    {
        Real3 center;
        auto bbox = bboxes_[indices[i].unchecked_get()];

        for (auto axis : range(Axis::size_))
        {
            auto ax = to_int(axis);
            center[ax] = (bbox.lower()[ax] + bbox.upper()[ax]) / 2;
        }
        centers[i] = center;
    }
    return centers;
}

//---------------------------------------------------------------------------//
/*!
 * Create sorted and uniquified X, Y, Z values of bbox centers.
 */
BIHBuilder::AxesCenters BIHBuilder::axes_centers(VecReal3 const& centers) const
{
    CELER_EXPECT(!centers.empty());

    AxesCenters axes_centers{{}, {}, {}};

    for (auto center : centers)
    {
        for (auto axis : range(Axis::size_))
        {
            auto ax = to_int(axis);
            axes_centers[ax].push_back(center[ax]);
        }
    }

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        sort_and_uniquify(axes_centers[ax]);
    }

    return axes_centers;
}

//---------------------------------------------------------------------------//
/*!
 * Bounding box of a collection of bounding boxes.
 */
BoundingBox BIHBuilder::meta_bbox(VecIndices const& indices) const
{
    CELER_EXPECT(!indices.empty());

    auto inf = std::numeric_limits<real_type>::infinity();

    Real3 lower = {inf, inf, inf};
    Real3 upper = {-inf, -inf, -inf};

    for (auto id : indices)
    {
        auto bbox = bboxes_[id.unchecked_get()];
        for (auto axis : range(Axis::size_))
        {
            auto ax = to_int(axis);
            lower[ax] = std::min(lower[ax], bbox.lower()[ax]);
            upper[ax] = std::max(upper[ax], bbox.upper()[ax]);
        }
    }

    return {lower, upper};
}

//---------------------------------------------------------------------------//
/*!
 * Create a vector of axes sorted from longest to shortest.
 */
BIHBuilder::VecAxes BIHBuilder::sort_axes(BoundingBox const& bbox) const
{
    VecAxes axes;
    std::vector<real_type> lengths;

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        axes.push_back(axis);
        lengths.push_back(bbox.upper()[ax] - bbox.lower()[ax]);
    }

    std::sort(axes.begin(), axes.end(), [&](Axis axis1, Axis axis2) {
        return lengths[to_int(axis1)] > lengths[to_int(axis2)];
    });
    return axes;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
