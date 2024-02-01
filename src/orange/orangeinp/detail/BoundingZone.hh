//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BoundingZone.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/BoundingBox.hh"
#include "orange/BoundingBoxUtils.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Exterior and interior CSG node bounding boxes.
 *
 * This class partition space into "known inside" a volume, "maybe inside,
 * maybe outside", and "known outside".
 *
 * \verbatim
 *  outside           ; inside
 *   +-------------+  ;  +-------------+
 *   | maybe   ..  |  ;  | maybe   ..  |
 *   |  .......  . |  ;  |  .......  . |
 *   | .+-----+ .  |  ;  | .+-----+ .  |
 *   | .| in  |  ..|  ;  | .| out |  ..|
 *   | .|     |   .|  ;  | .|     |   .|
 *   | .+-----+ .. |  ;  | .+-----+ .. |
 *   | ........    |  ;  | ........    |
 *   +-------------+  ;  +-------------+
 *                    ;
 *   negated = false  ;  negated = true
 * \endverbatim
 *
 * - Known outside is the typical bounding box case: nowhere outside the
 *   bounding box is inside the volume; i.e., the box completely encloses the
 *   volume (but may also enclose parts that *aren't* volume).
 * - Known inside can be used for safety distance calculation.
 * - Flipping the two cases when applying a "not" operator to a CSG node: so
 *   "known inside" becomes "known outside" and could potentially be flipped
 *   back in a subsequent simplification.
 *
 * The \c exterior box always encloses (or is identical to) \c interior in the
 * bounding zone. If \c negated is \c false, then the exterior box is what we
 * typically think of as a bounding box. If \c true, then even though the
 * exterior bounding box encloses interior box, *all points inside
 * the interior box are known to be outside the volume*.  Regardless of
 * negation, the area between the interior and exterior bounding boxes is \em
 * indeterminate: a point in that space may be within the CSG region, or it may
 * not.
 *
 * The behavior of boundaries (whether coincident boundary/shape should count
 * as being "interior" or "exterior") shouldn't matter because we will always
 * "bump" boundaries before using them for transport.
 *
 * The following table shows the semantics of set operations on the bounding
 * zones.  Below, \c ~zone means a zone with the \c negated flag set; \c ~box
 * means "complement of a bbox": all space *except*
 * inside box; and the union operator implies the (possibly disconnected and
 * probably not box-shaped!) union of the two boxes. \c A_i is the interior
 * (enclosed) box for A, and \c A_x is the exterior (enclosing) box.
 *
 * | Zone      | known inside   | known outside   |
 * | ------    | ------------   | -------------   |
 * | `A`       | `A_i`          | `~A_x`          |
 * | `~A`      | `~A_x`         | `A_i`           |
 * | `A & B`   | `A_i & B_i`    | `~(A_x & B_x)`  |
 * | `~A & B ` | `~A_x & B_i`   | `~(~A_i & B_x)` |
 * | `~A & ~B` | `~A_x & ~B_x`  | `~(~A_i & ~B_i)`|
 * | `A | B`   | `A_i | B_i`    | `~(A_x | B_x)`  |
 * | `~A | B ` | `~A_x | B_i`   | `~(~A_i | B_x)` |
 * | `~A | ~B` | `~A_x | ~B_x`  | `~(~A_i | ~B_i)`|
 *
 * The following set algebra can be used:
 * - Involution: \verbatim ~~A <=> A \endverbatim
 * - De Morgan's law 1: \verbatim ~A | ~B <=> ~(A & B) \endverbatim
 * - De Morgan's law 1:\verbatim ~A & ~B <=> ~(A | B) \endverbatim
 * - Set difference: \verbatim A & ~B <=> A - B \endverbatim
 * - Negated set difference: \verbatim A | ~B <=> ~(B - A) \endverbatim
 *
 * The default bounding zone places all points in the \em indeterminate zone:
 * the exterior is the "universe set" (infinite) and the interior is the "empty
 * set" (null).
 */
struct BoundingZone
{
    using BBox = ::celeritas::BoundingBox<>;

    BBox interior;
    BBox exterior = BBox::from_infinite();
    bool negated{false};  //!< "exterior" means "known inside"

    // Flip inside and outside
    inline void negate();
};

// Calculate the intersection of two bounding zones
BoundingZone calc_intersection(BoundingZone const& a, BoundingZone const& b);

// Calculate the union of two bounding zones
BoundingZone calc_union(BoundingZone const& a, BoundingZone const& b);

//---------------------------------------------------------------------------//
/*!
 * Flip inside and outside.
 *
 * This doesn't swap bounding boxes because the "exterior" box still is larger
 * than the interior.
 */
void BoundingZone::negate()
{
    CELER_EXPECT(!this->exterior || !this->interior
                 || encloses(this->exterior, this->interior));
    this->negated = !this->negated;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
