//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrientedBoundingZone.cc
//---------------------------------------------------------------------------//
#include "OrientedBoundingZone.hh"

#include <numeric>

#include "orange/transform/TransformVisitor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from inner/outer half-widths and a corresponding transform.
 */
OrientedBoundingZone::OrientedBoundingZone(Real3Id inner_hw_id,
                                           Real3Id outer_hw_id,
                                           TransformId transform_id,
                                           Storage* storage)
    : inner_hw_id_(inner_hw_id)
    , outer_hw_id_(outer_hw_id)
    , transform_id_(transform_id)
    , storage_(storage)
{
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance for a position inside the outer box.
 *
 * There are two cases:
 *
 * Case 1: the point is between the inner and outer boxes, resulting in a
 * safety distance of zero.
 *
 * Case 2: the point is inside both the inner and outer boxes, in which case
 * the safety distance is the minimum distance from the given point to any
 * point on the outer box. This is calculated by finding in the minimum of the
 * distances to each half width.
 */
real_type OrientedBoundingZone::safety_distance_inside(Real3 pos)
{
    CELER_EXPECT(this->is_inside_outer(pos));

    auto trans_pos = this->quadrant_zero(this->translate(pos));
    auto inner_hw = storage_->half_widths[inner_hw_id_];

    if (!this->is_inside(trans_pos, inner_hw))
    {
        return 0.;
    }

    auto outer_hw = storage_->half_widths[outer_hw_id_];
    return std::accumulate(
        range(Axis::size_).begin(),
        range(Axis::size_).end(),
        std::numeric_limits<real_type>::infinity(),
        [&](real_type min_dist, auto ax) {
            return std::min(min_dist,
                            outer_hw[to_int(ax)] - trans_pos[to_int(ax)]);
        });
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance for any position outside the inner box.
 *
 * There are two cases:
 *
 * Case 1: the point is between the inner and outer boxes, resulting in a
 * safety distance of zero.
 *
 * Case 2: the point is outside both the inner and outer boxes, in which case
 * the safety distance is the minimum distance from the given point to any
 * point on the inner box. This can be calculated as:
 *
 * d = sqrt(max(0, px - hx)^2 + max(0, py - hy)^2 + max(0, pz - hz)^2)
 *
 * for a point in quadrant zero at (px, py, pz) and a box with half-widths (hx,
 * hy, hz).
 */
real_type OrientedBoundingZone::safety_distance_outside(Real3 pos)
{
    CELER_EXPECT(!this->is_inside_inner(pos));

    auto trans_pos = this->quadrant_zero(this->translate(pos));
    auto outer_hw = storage_->half_widths[outer_hw_id_];

    if (this->is_inside(trans_pos, outer_hw))
    {
        return 0.;
    }

    auto inner_hw = storage_->half_widths[inner_hw_id_];
    Real3 maxes;
    std::transform(range(Axis::size_).begin(),
                   range(Axis::size_).end(),
                   maxes.begin(),
                   [&](auto ax) {
                       return std::max(
                           real_type{0},
                           trans_pos[to_int(ax)] - inner_hw[to_int(ax)]);
                   });

    return std::sqrt(
        std::inner_product(maxes.begin(), maxes.end(), maxes.begin(), 0.0));
}

//---------------------------------------------------------------------------//
/*!
 * Determine if a position is inside the inner box.
 */
bool OrientedBoundingZone::is_inside_inner(Real3 const& pos)
{
    return this->is_inside(this->translate(pos),
                           storage_->half_widths[inner_hw_id_]);
}

//---------------------------------------------------------------------------//
/*!
 * Determine if a position is inside the outer box.
 */
bool OrientedBoundingZone::is_inside_outer(Real3 const& pos)
{
    return this->is_inside(this->translate(pos),
                           storage_->half_widths[outer_hw_id_]);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Translate a position to the OBZ coordinate system.
 */
Real3 OrientedBoundingZone::translate(Real3 const& pos)
{
    Real3 trans_pos;
    TransformVisitor apply_transform(storage_->transforms, storage_->reals);
    auto transform_down
        = [&pos, &trans_pos](auto&& t) { trans_pos = t.transform_down(pos); };

    apply_transform(transform_down, transform_id_);
    return trans_pos;
}

//---------------------------------------------------------------------------//
/*!
 * Reflect a position into quadrant zero.
 */
Real3 OrientedBoundingZone::quadrant_zero(Real3 const& pos)
{
    auto temp = pos;

    std::transform(temp.begin(), temp.end(), temp.begin(), [](real_type x) {
        return std::abs(x);
    });

    return temp;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if a translated position is inside a translated box.
 *
 * The box is assumed to have the translation of transform_id_ with the given
 * half-widths.
 *
 * This function takes a position that has already been translated into the
 * local coordinate system of the OBZ to prevent cases where translations
 * are performed multiple times for the same given point.
 */
bool OrientedBoundingZone::is_inside(Real3 const& trans_pos,
                                     Real3 const& half_widths)
{
    return std::all_of(
        range(Axis::size_).begin(), range(Axis::size_).end(), [&](auto ax) {
            return trans_pos[to_int(ax)] <= half_widths[to_int(ax)];
        });
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
