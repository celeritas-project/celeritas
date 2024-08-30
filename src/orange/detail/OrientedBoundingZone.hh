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
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
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
    template<class T>
    using StorageItems
        = Collection<T, Ownership::const_reference, MemSpace::native>;

    using FastReal3 = OrientedBoundingZoneRecord::Real3;
    using FastReal3Id = OrientedBoundingZoneRecord::Real3Id;
    using fast_real_type = FastReal3::value_type;

    struct StoragePointers
    {
        StorageItems<FastReal3> const* half_widths;
        StorageItems<TransformRecord> const* transforms;
        StorageItems<real_type> const* reals;

        operator bool() const { return half_widths && transforms && reals; }
    };
    //!@}

  public:
    // Construct from inner/outer half-widths and a corresponding transform
    inline CELER_FUNCTION
    OrientedBoundingZone(OrientedBoundingZoneRecord const& obz_record,
                         StoragePointers const& sp);

    // Calculate the safety distance for any position inside the outer box
    inline CELER_FUNCTION real_type calc_safety_inside(Real3 const& pos);

    // Calculate the safety distance for any position outside the inner box
    inline CELER_FUNCTION real_type calc_safety_outside(Real3 const& pos);

    // Determine the sense of position with respect to the bounding zone
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos);

  private:
    // >> DATA
    OrientedBoundingZoneRecord const& obz_record_;
    StoragePointers sp_;

    //// HELPER METHODS ////

    // Translate a position into the OBZ coordinate system
    inline CELER_FUNCTION Real3 translate(Real3 const& pos);

    // Reflect a position into quadrant zero
    inline CELER_FUNCTION Real3 quadrant_zero(Real3 const& pos);

    // Determine if a translated position is inside a translated box
    inline CELER_FUNCTION bool
    is_inside(Real3 const& trans_pos, FastReal3 const& half_widths);

    // Get half-widths for a given index
    inline CELER_FUNCTION FastReal3 get_hw(FastReal3Id hw_id);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from inner/outer half-widths and a corresponding transform.
 */
CELER_FUNCTION
OrientedBoundingZone::OrientedBoundingZone(
    OrientedBoundingZoneRecord const& obz_record, StoragePointers const& sp)
    : obz_record_(obz_record), sp_(sp)
{
    CELER_EXPECT(obz_record_);
    CELER_EXPECT(sp_);
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
 * point on the outer box. This is calculated by finding the minimum of the
 * distances to each half width.
 */
CELER_FUNCTION real_type
OrientedBoundingZone::calc_safety_inside(Real3 const& pos)
{
    CELER_EXPECT(this->calc_sense(pos) != SignedSense::outside);

    auto trans_pos = this->quadrant_zero(this->translate(pos));

    if (!this->is_inside(trans_pos, this->get_hw(obz_record_.inner_hw_id)))
    {
        // Case 1: between inner and outer boxes
        return 0;
    }

    // Case 2: outside outer box
    auto outer_hw = this->get_hw(obz_record_.outer_hw_id);

    fast_real_type min_dist = numeric_limits<real_type>::infinity();
    for (auto ax : range(Axis::size_))
    {
        min_dist = celeritas::min(
            min_dist,
            outer_hw[int(ax)]
                - static_cast<fast_real_type>(trans_pos[to_int(ax)]));
    }

    return static_cast<real_type>(min_dist);
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
CELER_FUNCTION real_type
OrientedBoundingZone::calc_safety_outside(Real3 const& pos)
{
    CELER_EXPECT(this->calc_sense(pos) != SignedSense::inside);

    auto trans_pos = this->quadrant_zero(this->translate(pos));
    auto outer_hw = this->get_hw(obz_record_.outer_hw_id);

    if (this->is_inside(trans_pos, outer_hw))
    {
        // Case 1: between inner and outer boxes
        return 0;
    }

    // Case 2: outside outer box
    auto inner_hw = this->get_hw(obz_record_.inner_hw_id);

    fast_real_type min_squared = 0;
    for (auto ax : range(Axis::size_))
    {
        auto temp
            = celeritas::max(fast_real_type{0},
                             static_cast<fast_real_type>(trans_pos[to_int(ax)])
                                 - inner_hw[to_int(ax)]);
        min_squared += ipow<2>(temp);
    }

    return sqrt(min_squared);
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of position with respect to the OBZ.
 *
 * If the position is between the inner and outer bounding box its sense is
 * SignedSense::on.
 */
CELER_FUNCTION SignedSense OrientedBoundingZone::calc_sense(Real3 const& pos)
{
    auto trans_pos = this->translate(pos);

    if (this->is_inside(trans_pos, this->get_hw(obz_record_.inner_hw_id)))
    {
        return SignedSense::inside;
    }
    else if (this->is_inside(trans_pos, this->get_hw(obz_record_.outer_hw_id)))
    {
        return SignedSense::on;
    }
    else
    {
        return SignedSense::outside;
    }
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Translate a position to the OBZ coordinate system.
 */
CELER_FUNCTION Real3 OrientedBoundingZone::translate(Real3 const& pos)
{
    Real3 trans_pos;
    TransformVisitor apply_transform(*sp_.transforms, *sp_.reals);
    auto transform_down
        = [&pos, &trans_pos](auto&& t) { trans_pos = t.transform_down(pos); };
    apply_transform(transform_down, obz_record_.transform_id);

    return trans_pos;
}

//---------------------------------------------------------------------------//
/*!
 * Reflect a position into quadrant zero.
 */
CELER_FUNCTION Real3 OrientedBoundingZone::quadrant_zero(Real3 const& pos)
{
    Real3 temp;
    for (auto ax : range(Axis::size_))
    {
        temp[to_int(ax)] = std::fabs(pos[to_int(ax)]);
    }

    return temp;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if a translated position is inside a translated box.
 *
 * The box is assumed to have the translation of obz_record_.transform_id with
 * the given half-widths.
 *
 * This function takes a position that has already been translated into the
 * local coordinate system of the OBZ to prevent cases where translations
 * are performed multiple times for the same given point.
 */
CELER_FUNCTION bool
OrientedBoundingZone::is_inside(Real3 const& trans_pos,
                                FastReal3 const& half_widths)
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
/*!
 * Get half-widths for a given index.
 */
CELER_FUNCTION OrientedBoundingZone::FastReal3
OrientedBoundingZone::get_hw(FastReal3Id hw_id)
{
    CELER_EXPECT(hw_id < sp_.half_widths->size());
    return (*sp_.half_widths)[hw_id];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
