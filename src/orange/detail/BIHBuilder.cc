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

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sort and uniquify a vector.
 */
void sort_and_uniquify(std::vector<double>& vec)
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
BIHBuilder::BIHBuilder(VecBBox bboxes, BIHBuilder::Storage* storage)
    : bboxes_(bboxes), storage_(storage)
{
    // Check that we have at least two bounding boxes, with the first being
    // the exterior volume
    CELER_EXPECT(bboxes.size() >= 2);
    CELER_EXPECT(this->check_bbox_extents());
}

//---------------------------------------------------------------------------//
/*!
 * Create BIH Nodes.
 */
BIHBuilder::VecNodes BIHBuilder::operator()()
{
    // Create a vector of indices, excluding index 0 (i.e., the exterior)
    VecIndices indices(bboxes_.size() - 1);
    size_type id = 0;
    std::generate(
        indices.begin(), indices.end(), [&] { return LocalVolumeId{++id}; });

    VecNodes nodes;
    this->construct_tree(indices, nodes);

    return nodes;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct BIH nodes for a vector of bbox indices.
 */
void BIHBuilder::construct_tree(VecIndices const& indices, VecNodes& nodes)
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
                = {this->meta_bbox(left_indices).upper()[ax],
                   this->meta_bbox(right_indices).lower()[ax]};

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
BIHBuilder::Partition
BIHBuilder::find_partition(VecIndices const& indices, VecReal3 const& centers)
{
    CELER_EXPECT(indices.size() == centers.size());

    auto mb = this->meta_bbox(indices);
    VecAxes sorted_axes = this->sort_axes(mb);
    auto axes_centers = this->axes_centers(centers);

    Axis partition_axis;
    double partition_location;

    for (Axis axis : sorted_axes)
    {
        auto ax = to_int(axis);

        if (axes_centers[ax].size() < 2)
        {
            continue;
        }
        else
        {
            partition_axis = axis;
            auto size = axes_centers[ax].size();

            partition_location
                = (axes_centers[ax][size / 2 - 1] + axes_centers[ax][size / 2])
                  / 2;
            break;
        }
    }
    return {partition_axis, partition_location};
}

//---------------------------------------------------------------------------//
/*!
 * Divide bboxes into left and right branches based on a partition.
 */
BIHBuilder::PairVecIndices
BIHBuilder::partition_bboxes(VecIndices const& indices,
                             VecReal3 const& centers,
                             Partition const& p)
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
void BIHBuilder::make_leaf(BIHNode& node, VecIndices const& indices)
{
    CELER_EXPECT(!node);
    CELER_EXPECT(!indices.empty());

    node.vol_ids
        = make_builder(storage_).insert_back(indices.begin(), indices.end());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the centers of each bounding box.
 */
BIHBuilder::VecReal3 BIHBuilder::centers(VecIndices const& indices)
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
BIHBuilder::AxesCenters BIHBuilder::axes_centers(VecReal3 const& centers)
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
BoundingBox BIHBuilder::meta_bbox(VecIndices const& indices)
{
    CELER_EXPECT(!indices.empty());

    auto inf = std::numeric_limits<double>::infinity();

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
BIHBuilder::VecAxes BIHBuilder::sort_axes(BoundingBox const& bbox)
{
    VecAxes axes;
    std::vector<double> lengths;

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
/*!
 * Check that only the first bounding box (i.e. exterior volume) is fully inf.
 */
bool BIHBuilder::check_bbox_extents()
{
    return this->fully_inf(bboxes_[0])
           && std::all_of(
               bboxes_.begin() + 1, bboxes_.end(), [this](BoundingBox bbox) {
                   return !this->fully_inf(bbox);
               });
}

//---------------------------------------------------------------------------//
/*!
 * Check if a bounding box spans (-inf, inf) in every direction.
 */
bool BIHBuilder::fully_inf(BoundingBox const& bbox)
{
    auto max_double = std::numeric_limits<double>::max();

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        if (bbox.lower()[ax] > -max_double || bbox.upper()[ax] < max_double)
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
