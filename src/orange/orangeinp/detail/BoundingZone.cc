//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BoundingZone.cc
//---------------------------------------------------------------------------//
#include "BoundingZone.hh"

#include "orange/BoundingBoxUtils.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Whether to reduce or expand a bbox operation to enclose unknown space
enum class BoxOp : bool
{
    shrink,
    grow
};

//---------------------------------------------------------------------------//
// For now, be very conservative by returning infinities unless null
BBox calc_difference(BBox const& a, BBox const& b, BoxOp op)
{
    if (!b)
    {
        return a;
    }
    if (encloses(a, b))
    {
        return (op == BoxOp::shrink ? b : a);
    }
    if (encloses(b, a))
    {
        return BBox{};
    }
    return (op == BoxOp::shrink ? BBox{} : BBox::from_infinite());
}

//---------------------------------------------------------------------------//
// For now, be conservative by "shrinking" into the largest known box shape
BBox calc_union(BBox const& a, BBox const& b, BoxOp op)
{
    if (op == BoxOp::grow)
    {
        // Result encloses both and it can enclose space not in the original
        // two bboxes, so use standard function
        return calc_union(a, b);
    }

    // Union of A with null is A
    if (!a)
    {
        return b;
    }
    if (!b)
    {
        return a;
    }

    // Choose the larger box since the resulting box has to be strictly
    // enclosed by the space in the input boxes
    return calc_volume(a) > calc_volume(b) ? a : b;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Create an "everything is known inside" zone for intersecting.
 */
BoundingZone BoundingZone::from_infinite()
{
    return {BBox::from_infinite(), BBox::from_infinite(), false};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the intersection of two bounding zones.
 *
 * Here are the zones that result from intersections of two zones with
 * different negations:
 *
 * | Input     | Interior     | Exterior    | Negated  |
 * | ------    | ------------ | ----------- | -------- |
 * | `A & B`   | `A_i & B_i`  | `A_x & B_x` | false    |
 * | `A & ~B`  | `A_i - B_x`  | `A_x - B_i` | false    |
 * | `~A & B ` | `B_i - A_x`  | `B_x - A_i` | false    |
 * | `~A & ~B` | `A_i | B_i`  | `A_x | B_x` | true     |
 *
 * The above algebra for unions and intersections does *not* necessarily
 * produce boxes: it can produce a single box, or an orthogonal polyhedron
 * (having only right angles), or two disconnected boxes.
 * If the intersected regions are not boxes (and irregularly shaped regions are
 * always in the between zone):
 * - the interior result has to "shrink" to be completely enclosed by the
 *   resulting region, and
 * - the exterior has to "grow" to completely enclose the resulting region
 *   (i.e. it should be the bounding box of the resulting polyhedron).
 *
 * \todo Only under certain circumstances will unions and subtractions between
 * boxes result in an actual box shape. To be conservative, for now we return
 * an indeterminate zone for anything but intersection of two non-negated
 * zones.
 */
BoundingZone calc_intersection(BoundingZone const& a, BoundingZone const& b)
{
    BoundingZone result;
    result.negated = false;
    if (!a.negated && !b.negated)
    {
        // A & B
        result.interior = calc_intersection(a.interior, b.interior);
        result.exterior = calc_intersection(a.exterior, b.exterior);
    }
    else if (!a.negated && b.negated)
    {
        // A - B
        result.interior
            = calc_difference(a.interior, b.exterior, BoxOp::shrink);
        result.exterior = calc_difference(a.exterior, b.interior, BoxOp::grow);
    }
    else if (!b.negated && a.negated)
    {
        // B - A
        result.interior
            = calc_difference(b.interior, a.exterior, BoxOp::shrink);
        result.exterior = calc_difference(b.exterior, a.interior, BoxOp::grow);
    }
    else if (a.negated && b.negated)
    {
        // ~(A | B)
        result.interior = calc_union(a.interior, b.interior, BoxOp::shrink);
        result.exterior = calc_union(a.exterior, b.exterior, BoxOp::grow);
        result.negated = true;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the union of two bounding zones.
 *
 * Here are the zones that result from unioning of two zones with
 * different negations:
 *
 * | Input     | Interior     | Exterior     | Negated  |
 * | ------    | ------------ | ------------ | -------- |
 * | `A | B`   | `A_i | B_i`  | `A_x | B_x`  | false    |
 * | `A | ~B`  | `B_i - A_x`  | `B_x - A_i`  | true     |
 * | `~A | B ` | `A_i - B_x`  | `A_x - B_i`  | true     |
 * | `~A | ~B` | `A_i & B_i`  | `A_x & B_x`  | true     |
 *
 * As with the intersection, the interior has to shrink and the exterior has to
 * grow if the unioned regions aren't boxes.
 */
BoundingZone calc_union(BoundingZone const& a, BoundingZone const& b)
{
    BoundingZone result;
    result.negated = true;
    if (!a.negated && !b.negated)
    {
        // A | B
        result.interior = calc_union(a.interior, b.interior, BoxOp::shrink);
        result.exterior = calc_union(a.exterior, b.exterior, BoxOp::grow);
        result.negated = false;
    }
    else if (!a.negated && b.negated)
    {
        // ~(B - A)
        result.interior
            = calc_difference(a.interior, b.exterior, BoxOp::shrink);
        result.exterior = calc_difference(a.exterior, b.interior, BoxOp::grow);
    }
    else if (!b.negated && a.negated)
    {
        // ~(A - B)
        result.interior
            = calc_difference(b.interior, a.exterior, BoxOp::shrink);
        result.exterior = calc_difference(b.exterior, a.interior, BoxOp::grow);
    }
    else if (a.negated && b.negated)
    {
        // !(A & B)
        result.interior = calc_intersection(a.interior, b.interior);
        result.exterior = calc_intersection(a.exterior, b.exterior);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
