//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
enum class Zone
{
    interior,
    exterior,
    size_
};

//---------------------------------------------------------------------------//
//! Whether a bounding box is finite, null, or infinite; used for printing
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
// TODO: include tolerance in these calculations since the edge cases are weird
BBox calc_difference(BBox const& a, BBox const& b, Zone which)
{
    if (!b)
    {
        // Subtracting nothing: return early to avoid 'encloses' error
        return a;
    }
    if (which == Zone::interior)
    {
        if (encloses(b, a))
        {
            if (encloses(a, b))
            {
                // Edge case: a == b
                return a;
            }

            // The two "known inside" regions do not overlap: exactly null
            return {};
        }
        // Irregular region: conservatively return null
        return {};
    }
    else if (which == Zone::exterior)
    {
        // NOTE: we could return an exact null if `encloses(b, a)`
        //  *and not* `encloses(a, b)`, where the edge case of a = b must be
        //  considered.
        if (encloses(b, a))
        {
            if (encloses(a, b))
            {
                // Edge case: a == b
                return a;
            }
            // Never inside B and never outside A -> nowhere
            // *excluding* the edge case of a == b
            // (Should be rare in practice since this would be literally a null
            // region in space)
            return {};
        }

        // "Never" is a union of the negative exterior of A and the
        // interior of B; so an exterior bbox of A is conservative
        return a;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
BBox calc_union(BBox const& a, BBox const& b, Zone which)
{
    if (which == Zone::exterior)
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
 * boxes result in an actual box shape. The resulting bounding zone must
 * carefully respect the intermediate region.
 */
BoundingZone calc_intersection(BoundingZone const& a, BoundingZone const& b)
{
    BoundingZone result;
    if (!a.negated && !b.negated)
    {
        // A & B
        result.interior = calc_intersection(a.interior, b.interior);
        result.exterior = calc_intersection(a.exterior, b.exterior);
        result.negated = false;
    }
    else if (!a.negated && b.negated)
    {
        // A - B
        result.interior
            = calc_difference(a.interior, b.exterior, Zone::interior);
        result.exterior
            = calc_difference(a.exterior, b.interior, Zone::exterior);
        result.negated = false;
    }
    else if (!b.negated && a.negated)
    {
        // B - A
        result.interior
            = calc_difference(b.interior, a.exterior, Zone::interior);
        result.exterior
            = calc_difference(b.exterior, a.interior, Zone::exterior);
        result.negated = false;
    }
    else if (a.negated && b.negated)
    {
        // ~(A | B)
        result.interior = calc_union(a.interior, b.interior, Zone::interior);
        result.exterior = calc_union(a.exterior, b.exterior, Zone::exterior);
        result.negated = true;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the union of two bounding zones.
 *
 * We use DeMorgan's law to represent, e.g., `A | ~B` as `~(B - A)`.
 */
BoundingZone calc_union(BoundingZone const& a, BoundingZone const& b)
{
    BoundingZone result;
    if (!a.negated && !b.negated)
    {
        // A | B
        result.interior = calc_union(a.interior, b.interior, Zone::interior);
        result.exterior = calc_union(a.exterior, b.exterior, Zone::exterior);
        result.negated = false;
    }
    else if (!a.negated && b.negated)
    {
        // A | ~B = ~(~A & B) = ~(B - A)
        result.interior
            = calc_difference(b.interior, a.exterior, Zone::interior);
        result.exterior
            = calc_difference(b.exterior, a.interior, Zone::exterior);
        result.negated = true;
    }
    else if (!b.negated && a.negated)
    {
        // ~A | B = ~(A & ~B) = ~(A - B)
        result.interior
            = calc_difference(a.interior, b.exterior, Zone::interior);
        result.exterior
            = calc_difference(a.exterior, b.interior, Zone::exterior);
        result.negated = true;
    }
    else if (a.negated && b.negated)
    {
        // !(A & B)
        result.interior = calc_intersection(a.interior, b.interior);
        result.exterior = calc_intersection(a.exterior, b.exterior);
        result.negated = true;
    }
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
 * In this table, interior and exterior are abbreviated I and X. Note that the
 * interior box should \em always be enclosed by the exterior box (which is the
 * BZ's operator bool).
 *
 * Negated | [I]nterior | E[X]terior | Result
 * ------- | --------   | ---------- | -------
 * No      | Null       | Null       | Nowhere
 * No      | Null       | Finite     | Never outside X
 * No      | Null       | Infinite   | Maybe anywhere
 * No      | Finite     | Finite     | Always inside I, never outside X
 * No      | Finite     | Infinite   | Always inside I
 * No      | Infinite   | Infinite   | Everywhere
 * Yes     | Null       | Null       | Everywhere
 * Yes     | Null       | Finite     | Always outside X
 * Yes     | Null       | Infinite   | Maybe anywhere
 * Yes     | Finite     | Finite     | Always outside X, never inside I
 * Yes     | Finite     | Infinite   | Never inside I
 * Yes     | Infinite   | Infinite   | Nowhere
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
