//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BoundingZone.cc
//---------------------------------------------------------------------------//
#include "BoundingZone.hh"

#include "orange/BoundingBoxUtils.hh"

#if 1
#    include <iostream>
using std::cout;
using std::endl;
#endif

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

char const* to_cstring(BoxOp bo)
{
    switch (bo)
    {
        case BoxOp::shrink:
            return "shrink";
        case BoxOp::grow:
            return "grow";
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
//! Whether a bounding box is finite, null, or infinite
enum class BoxExtent
{
    null,
    finite,
    infinite
};

BoxExtent get_extent(BBox const& b)
{
    if (!b)
        return BoxExtent::null;
    if (is_infinite(b))
        return BoxExtent::infinite;
    return BoxExtent::finite;
}

//---------------------------------------------------------------------------//
// For now, be very conservative by returning infinities unless null
BBox calc_difference(BBox const& a, BBox const& b, BoxOp op)
{
    cout << "      + Subtract a=" << a << " - b=" << b << " ("
         << to_cstring(op) << ") -> ";
    if (!b)
    {
        cout << "a\n";
        return a;
    }
    if (encloses(a, b))
    {
        cout << (op == BoxOp::shrink ? "b" : "a") << "\n";
        return (op == BoxOp::shrink ? b : a);
    }
    if (encloses(b, a))
    {
        cout << "null\n";
        return BBox{};
    }
    cout << (op == BoxOp::shrink ? "null" : "inf") << "\n";
    return (op == BoxOp::shrink ? BBox{} : BBox::from_infinite());
}

//---------------------------------------------------------------------------//
// For now, be conservative by "shrinking" into the largest known box shape
BBox calc_union(BBox const& a, BBox const& b, BoxOp op)
{
    cout << "      + Union a=" << a << " - b=" << b << " (" << to_cstring(op)
         << ") -> ";

    if (op == BoxOp::grow)
    {
        cout << "union\n";
        // Result encloses both and it can enclose space not in the original
        // two bboxes, so use standard function
        return calc_union(a, b);
    }

    // Union of A with null is A
    if (!a)
    {
        cout << "b\n";
        return b;
    }
    if (!b)
    {
        cout << "a\n";
        return a;
    }

    // Choose the larger box since the resulting box has to be strictly
    // enclosed by the space in the input boxes
    cout << "larger\n";
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
    cout << "  - Intersect " << a << "\n    & " << b << ":\n";

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
    cout << "    -> " << result << endl;
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
    cout << "  - Union " << a << "\n    | " << b << ":\n";

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
    cout << "    -> " << result << endl;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get an infinite bbox if "negated", else get the exterior.
 */
BBox get_exterior_bbox(BoundingZone const& bz)
{
    if (bz.negated)
    {
        // Everything "outside" a finite region: infinite
        return BBox::from_infinite();
    }
    return bz.exterior;
}

//---------------------------------------------------------------------------//
/*!
 * Print for debugging.
 *
 * Negated | Interior | Exterior  | Result
 * ------- | -------- | --------- | -------
 * No      | Null     | Null      | Nowhere
 * No      | Null     | Finite    | Never outside X
 * No      | Null     | Infinite  | Maybe anywhere
 * No      | Finite   | Finite    | Always inside I, never outside X
 * No      | Finite   | Infinite  | Always inside I
 * No      | Infinite | Infinite  | Everywhere
 * Yes     | Null     | Null      | Everywhere
 * Yes     | Null     | Finite    | Always outside X
 * Yes     | Null     | Infinite  | Maybe anywhere
 * Yes     | Finite   | Finite    | Always outside X, never inside I
 * Yes     | Finite   | Infinite  | Never inside I
 * Yes     | Infinite | Infinite  | Nowhere
 */
std::ostream& operator<<(std::ostream& os, BoundingZone const& bz)
{
    CELER_EXPECT(bz);
    using BE = BoxExtent;
    BE const ibe = get_extent(bz.interior);
    BE const xbe = get_extent(bz.exterior);
    bool const neg = bz.negated;

    os << '{';
    if ((!neg && xbe == BE::null) || (neg && ibe == BE::infinite))
    {
        os << "nowhere";
    }
    else if ((!neg && ibe == BE::infinite) || (neg && xbe == BE::null))
    {
        os << "everywhere";
    }
    else if (ibe == BE::null && xbe == BE::infinite)
    {
        os << "maybe anywhere";
    }
    else
    {
        bool print_and{false};
        if (ibe != BE::null)
        {
            os << (neg ? "never" : "always") << " inside " << bz.interior;
            print_and = true;
        }
        if (xbe != BE::infinite)
        {
            if (print_and)
            {
                os << " and ";
            }
            os << (neg ? "always" : "never") << " outside " << bz.exterior;
        }
    }
    os << '}';

    return os;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
