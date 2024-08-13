//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrientedBoundingZone.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeData.hh"
#include "orange/transform/TransformVisitor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Oriented bounding zone for saftey distance calculations.
 *
 * Here, the bounding zone is defined by an inner and outer bounding box
 * (specified via half-widths) transformed by the same transformation;
 * pressumably the transformation of the volume they correspond to. The
 * safety distance can be approximated as the minimum distance to the inner or
 * outer box. For points lying between the inner and outer boxes, the safety
 * distance is zero.
 */
class OrientedBoundingZone
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<OrientedBoundingZoneData>;
    //!@}

  public:
    // Construct from inner/outer half-widths and a corresponding transform
    inline CELER_FUNCTION OrientedBoundingZone(Real3Id inner_hw_id,
                                               Real3Id outer_hw_id,
                                               TransformId transform_id,
                                               Storage* storage);

    // Calculate the safety distance for any position inside the outer box
    inline CELER_FUNCTION real_type safety_distance_inside(Real3 pos);

    // Calculate the safety distance for any position outside the inner box
    inline CELER_FUNCTION real_type safety_distance_outside(Real3 pos);

    // Determine if a position is inside the inner box
    inline CELER_FUNCTION bool is_inside_inner(Real3 const& pos);

    // Determine if a position is inside the outer box
    inline CELER_FUNCTION bool is_inside_outer(Real3 const& pos);

  private:
    // >> DATA
    Real3Id inner_hw_id_;
    Real3Id outer_hw_id_;
    TransformId transform_id_;
    Storage* storage_;

    //// HELPER METHODS ////

    // Translate a position into the OBZ coordinate system
    inline CELER_FUNCTION Real3 translate(Real3 const& pos);

    // Reflect a position into quadrant zero
    inline CELER_FUNCTION Real3 quadrant_zero(Real3 const& pos);

    // Determine if a translated position is inside a translated box
    inline CELER_FUNCTION bool
    is_inside(Real3 const& trans_pos, Real3 const& half_widths);
};

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
        return 0;
    }

    auto outer_hw = storage_->half_widths[outer_hw_id_];

    real_type min_dist = numeric_limits<real_type>::infinity();
    for (auto ax : range(Axis::size_))
    {
        min_dist = celeritas::min(min_dist,
                                  outer_hw[int(ax)] - trans_pos[to_int(ax)]);
    }

    return min_dist;
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
 * \f[
 * d = \sqrt(\max(0, p_x - h_x)^2 + max(0, p_y - h_y)^2 + max(0, p_z - h_z)^2)
 * \f]
 *
 * for a point in quadrant zero at (\em p_x, \em p_y, \em p_z) and a box with
 * half-widths (\em h_x, \em h_y, \em h_z).
 */
real_type OrientedBoundingZone::safety_distance_outside(Real3 pos)
{
    CELER_EXPECT(!this->is_inside_inner(pos));

    auto trans_pos = this->quadrant_zero(this->translate(pos));
    auto outer_hw = storage_->half_widths[outer_hw_id_];

    if (this->is_inside(trans_pos, outer_hw))
    {
        return 0;
    }

    auto inner_hw = storage_->half_widths[inner_hw_id_];

    real_type min_squared = 0;
    for (auto ax : range(Axis::size_))
    {
        auto temp = celeritas::max(
            real_type{0}, trans_pos[to_int(ax)] - inner_hw[to_int(ax)]);
        min_squared += temp * temp;
    }

    return sqrt(min_squared);
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
    Real3 temp;

    for (auto ax : range(Axis::size_))
    {
        temp[to_int(ax)] = abs(pos[to_int(ax)]);
    }

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
    for (auto ax : range(Axis::size_))
    {
        if (trans_pos[to_int(ax)] > half_widths[to_int(ax)])
        {
            return false;
        }
    }

    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
