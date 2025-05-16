//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrientedBoundingZone.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"

#include "../OrangeData.hh"
#include "../OrangeTypes.hh"
#include "../transform/TransformVisitor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*! Oriented bounding zone for safety distance calculations.
 *
 * Here, the oriented bounding zone (OBZ) is defined by inner and outer
 * bounding boxes transformed by the same transformation --- that of the volume
 * they correspond to. The OBZ bounding boxes are stored on the
 * OrientedBoundingZoneRecord as:
 *
 * 1. a vector of half-widths,
 *
 * 2. an additional *translation* object for each bounding box specifying how
 * the center of each bounding box is offset from the center of the OBZ
 * coordinate system.
 *
 * Consequently, points in the unit's coordinate system must be first
 * transformed by \c trans_id into the OBZ coordinate system (resulting in
 * "trans_pos"), then offset, i.e. translated, by \c inner_offset_id or
 * \c outer_offset_id into the inner or outer bbox coordinate system (resulting
 * in "offset_pos"). It is noted that these offset positions are always
 * automatically reflected into the first quadrant.
 *
 * The safety distance can be approximated as the minimum distance to the inner
 * or outer box. For points lying between the inner and outer boxes, the safety
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
    using fast_real_type = FastReal3::value_type;

    struct StoragePointers
    {
        StorageItems<TransformRecord> const* transforms;
        StorageItems<real_type> const* reals;

        operator bool() const { return transforms && reals; }
    };
    //!@}

  public:
    // Construct from an OBZ record and corresponding storage
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
    //!@{
    //! \name Type aliases

    //! Enum to distinguish between inner and outer bboxes
    enum class BBoxType
    {
        inner,  //!< inner bbox
        outer  //!< outer bbox
    };

    //! A position within the innner or outer bbox coordinate system
    struct OffsetPos
    {
        Real3 pos;  //!< the position
        BBoxType bbt;  //!< the bbox coordinate system
    };
    //!@}

    // >> DATA
    OrientedBoundingZoneRecord const& obz_record_;
    StoragePointers sp_;

    //// HELPER METHODS ////

    // Translate a position into the OBZ coordinate system
    inline CELER_FUNCTION Real3 translate(Real3 const& unit_pos);

    // Offset a pre-translated position into a bbox coordinate system
    inline CELER_FUNCTION OffsetPos apply_offset(Real3 const& trans_pos,
                                                 BBoxType bbt);

    // Determine if an offset position is inside its respective bbox
    inline CELER_FUNCTION bool is_inside(OffsetPos const& off_pos);

    // Get half-widths for a bbox
    inline CELER_FUNCTION FastReal3 get_hw(BBoxType bbt);

    // Reflect a position into quadrant one
    inline CELER_FUNCTION Real3 quadrant_one(Real3 const& pos);

    // Convert BBoxType enum value to int
    CELER_CONSTEXPR_FUNCTION int to_int(BBoxType bbt);
};

//---------------------------------------------------------------------------//
/*!
 * Construct from an OBZ record and corresponding storage.
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
 * point on the inner box. This is calculated by finding the minimum of the
 * distances to each half width.
 */
CELER_FUNCTION real_type
OrientedBoundingZone::calc_safety_inside(Real3 const& pos)
{
    CELER_EXPECT(this->calc_sense(pos) != SignedSense::outside);

    auto trans_pos = this->translate(pos);

    if (!this->is_inside(this->apply_offset(trans_pos, BBoxType::inner)))
    {
        // Case 1: between inner and outer boxes
        return 0;
    }

    // Case 2: inside outer box
    auto inner_offset_pos = this->apply_offset(trans_pos, BBoxType::inner);
    auto inner_hw = this->get_hw(BBoxType::inner);

    fast_real_type min_dist = numeric_limits<real_type>::infinity();
    for (auto ax : range(Axis::size_))
    {
        min_dist = celeritas::min(
            min_dist,
            inner_hw[int(ax)]
                - static_cast<fast_real_type>(
                    inner_offset_pos.pos[celeritas::to_int(ax)]));
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
 * point on the outer box. This can be calculated as:
 *
 * \f[
 * d = \sqrt(\max(0, p_x - h_x)^2 + max(0, p_y - h_y)^2 + max(0, p_z - h_z)^2)
 * \f]
 *
 * for a point in quadrant one at (\em p_x, \em p_y, \em p_z) and a box with
 * half-widths (\em h_x, \em h_y, \em h_z).
 */
CELER_FUNCTION real_type
OrientedBoundingZone::calc_safety_outside(Real3 const& pos)
{
    CELER_EXPECT(this->calc_sense(pos) != SignedSense::inside);

    auto trans_pos = this->translate(pos);

    if (this->is_inside(this->apply_offset(trans_pos, BBoxType::outer)))
    {
        // Case 1: between inner and outer boxes
        return 0;
    }

    // Case 2: outside outer box
    auto outer_offset_pos = this->apply_offset(trans_pos, BBoxType::outer);
    auto outer_hw = this->get_hw(BBoxType::outer);

    fast_real_type min_squared = 0;
    for (auto ax : range(Axis::size_))
    {
        auto temp
            = celeritas::max(fast_real_type{0},
                             static_cast<fast_real_type>(
                                 outer_offset_pos.pos[celeritas::to_int(ax)])
                                 - outer_hw[celeritas::to_int(ax)]);
        min_squared += ipow<2>(temp);
    }

    return sqrt(min_squared);
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of position with respect to the OBZ.
 *
 * If the position is between the inner and outer bboxes its sense is
 * SignedSense::on.
 */
CELER_FUNCTION SignedSense OrientedBoundingZone::calc_sense(Real3 const& pos)
{
    auto trans_pos = this->translate(pos);

    if (this->is_inside(this->apply_offset(trans_pos, BBoxType::inner)))
    {
        return SignedSense::inside;
    }
    else if (this->is_inside(this->apply_offset(trans_pos, BBoxType::outer)))
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
 * Translate a position into the OBZ coordinate system.
 */
CELER_FUNCTION Real3 OrientedBoundingZone::translate(Real3 const& pos)
{
    TransformVisitor apply_transform(*sp_.transforms, *sp_.reals);
    auto transform_down = [&pos](auto&& t) { return t.transform_down(pos); };

    return apply_transform(transform_down, obz_record_.trans_id);
}

//---------------------------------------------------------------------------//
/*!
 * Offset a pre-translated position into a bbox coordinate system.
 *
 * This function also reflects the point into quadrant one.
 */
CELER_FUNCTION auto
OrientedBoundingZone::apply_offset(Real3 const& trans_pos, BBoxType bbt)
    -> OffsetPos
{
    TransformVisitor apply_transform(*sp_.transforms, *sp_.reals);
    auto transform_down
        = [&trans_pos](auto&& t) { return t.transform_down(trans_pos); };

    return {this->quadrant_one(apply_transform(
                transform_down, obz_record_.offset_ids[this->to_int(bbt)])),
            bbt};
}

//---------------------------------------------------------------------------//
/*!
 * Determine if an offset position is inside its respective bbox.
 */
CELER_FUNCTION bool OrientedBoundingZone::is_inside(OffsetPos const& off_pos)
{
    auto const& half_widths = this->get_hw(off_pos.bbt);

    for (auto ax : range(Axis::size_))
    {
        if (off_pos.pos[celeritas::to_int(ax)]
            > half_widths[celeritas::to_int(ax)])
        {
            return false;
        }
    }

    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Get half-widths for a bbox.
 */
CELER_FUNCTION OrientedBoundingZone::FastReal3
OrientedBoundingZone::get_hw(BBoxType bbt)
{
    return obz_record_.half_widths[this->to_int(bbt)];
}

//---------------------------------------------------------------------------//
/*!
 * Reflect a position into quadrant one.
 */
CELER_FUNCTION Real3 OrientedBoundingZone::quadrant_one(Real3 const& pos)
{
    Real3 temp;
    for (auto ax : range(Axis::size_))
    {
        temp[celeritas::to_int(ax)] = std::fabs(pos[celeritas::to_int(ax)]);
    }
    return temp;
}

//---------------------------------------------------------------------------//
/*!
 * Convert BBoxType enum value to int.
 */
CELER_CONSTEXPR_FUNCTION int
OrientedBoundingZone::to_int(OrientedBoundingZone::BBoxType bbt)
{
    return static_cast<int>(bbt);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
