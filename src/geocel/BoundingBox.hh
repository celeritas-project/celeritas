//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BoundingBox.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/grid/GridTypes.hh"
#include "corecel/math/NumericLimits.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned bounding box.
 *
 * Bounding boxes "contain" all points inside \em and on their faces. See \c
 * is_inside in \c BoundingBoxUtils.hh .
 *
 * The default bounding box is "null", which has at least one \c lower
 * coordinate greater than its \c upper coordinate: it evaluates to \c false .
 * A null bounding box still has the ability to be unioned and intersected with
 * other bounding boxes with the expected effect, but geometrical operations on
 * it (center, surface area, volume) are prohibited.
 *
 * A "degenerate" bounding box is one that is well-defined but has zero volume
 * because at least one lower coordinate is equal to the corresponding upper
 * coordinate. Any point on the surface of this bounding box is still "inside".
 * It may have nonzero surface area but will have zero volume.
 */
template<class T = ::celeritas::real_type>
class BoundingBox
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using Real3 = Array<real_type, 3>;
    //!@}

  public:
    // Construct from infinite extents
    static inline CELER_FUNCTION BoundingBox from_infinite();

    // Construct from unchecked lower/upper bounds
    static CELER_CONSTEXPR_FUNCTION BoundingBox
    from_unchecked(Real3 const& lower, Real3 const& upper);

    // Construct in unassigned state
    CELER_CONSTEXPR_FUNCTION BoundingBox();

    // Construct from upper and lower points
    inline CELER_FUNCTION BoundingBox(Real3 const& lower, Real3 const& upper);

    //// ACCESSORS ////

    //! Lower bbox coordinate
    CELER_CONSTEXPR_FUNCTION Real3 const& lower() const
    {
        return this->point(Bound::lo);
    }

    //! Upper bbox coordinate
    CELER_CONSTEXPR_FUNCTION Real3 const& upper() const
    {
        return this->point(Bound::hi);
    }

    //! Access a bounding point
    CELER_CONSTEXPR_FUNCTION Real3 const& point(Bound b) const
    {
        return points_[to_int(b)];
    }

    // Whether the bbox is non-null
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const;

    //// MUTATORS ////

    // Reduce the bounding box's extent along an axis
    CELER_CONSTEXPR_FUNCTION void
    shrink(Bound bnd, Axis axis, real_type position);

    // Increase the bounding box's extent along an axis
    CELER_CONSTEXPR_FUNCTION void
    grow(Bound bnd, Axis axis, real_type position);

    // Increase the bounding box's extent on both bounds
    CELER_CONSTEXPR_FUNCTION void grow(Axis axis, real_type position);

  private:
    Array<Real3, 2> points_;  //!< lo/hi points

    // Implementation of 'from_unchecked' (true type 'tag')
    CELER_CONSTEXPR_FUNCTION
    BoundingBox(std::true_type, Real3 const& lower, Real3 const& upper);
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Bounding box for host metadata
using BBox = BoundingBox<>;

//---------------------------------------------------------------------------//
// INLINE FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Test equality of two bounding boxes.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator==(BoundingBox<T> const& lhs, BoundingBox<T> const& rhs)
{
    return lhs.lower() == rhs.lower() && lhs.upper() == rhs.upper();
}

//---------------------------------------------------------------------------//
/*!
 * Test inequality of two bounding boxes.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION bool
operator!=(BoundingBox<T> const& lhs, BoundingBox<T> const& rhs)
{
    return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
/*!
 * Determine if a point is contained in a bounding box.
 *
 * No point is ever contained in a null bounding box. A degenerate bounding
 * box will return "true" for any point on its face.
 */
template<class T, class U>
CELER_CONSTEXPR_FUNCTION bool
is_inside(BoundingBox<T> const& bbox, Array<U, 3> const& point)
{
    return bbox.lower()[0] <= point[0] && point[0] <= bbox.upper()[0]
           && bbox.lower()[1] <= point[1] && point[1] <= bbox.upper()[1]
           && bbox.lower()[2] <= point[2] && point[2] <= bbox.upper()[2];
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create a bounding box with infinite extents.
 */
template<class T>
CELER_FUNCTION BoundingBox<T> BoundingBox<T>::from_infinite()
{
    constexpr real_type inf = numeric_limits<real_type>::infinity();
    return {{-inf, -inf, -inf}, {inf, inf, inf}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box from unchecked lower/upper bounds.
 *
 * This should be used exclusively for utilities that understand the
 * "null" implementation of the bounding box.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>
BoundingBox<T>::from_unchecked(Real3 const& lo, Real3 const& hi)
{
    return BoundingBox<T>{std::true_type{}, lo, hi};
}

//---------------------------------------------------------------------------//
/*!
 * Create a null bounding box.
 *
 * This should naturally satisfy
 * \code
        calc_union(BBox{}, other) = other:
   \endcode
 *  and
 * \code
        calc_intersection(BBox{}, other) = other;
   \endcode
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::BoundingBox()
{
    constexpr real_type inf = numeric_limits<real_type>::infinity();
    points_[to_int(Bound::lo)] = {inf, inf, inf};
    points_[to_int(Bound::hi)] = {-inf, -inf, -inf};
}

//---------------------------------------------------------------------------//
/*!
 * Create a non-null bounding box from two points.
 *
 * The lower and upper points are allowed to be equal (an empty bounding box
 * at a single point) but upper must not be less than lower.
 */
template<class T>
CELER_FUNCTION BoundingBox<T>::BoundingBox(Real3 const& lo, Real3 const& hi)
    : points_{{lo, hi}}
{
    if constexpr (CELERITAS_DEBUG)
    {
        for (auto ax : {Axis::x, Axis::y, Axis::z})
        {
            CELER_EXPECT(this->lower()[to_int(ax)]
                         <= this->upper()[to_int(ax)]);
        }
    }
    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Create a possibly null bounding box from two points.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION
BoundingBox<T>::BoundingBox(std::true_type, Real3 const& lo, Real3 const& hi)
    : points_{{lo, hi}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox is in a valid state.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::operator bool() const
{
    return this->lower()[0] <= this->upper()[0]
           && this->lower()[1] <= this->upper()[1]
           && this->lower()[2] <= this->upper()[2];
}

//---------------------------------------------------------------------------//
/*!
 * Reduce (clip) the bounding box's extent along an axis.
 *
 * If the point is inside the box, the box is clipped so the given boundary is
 * on that point. Otherwise no change is made.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION void
BoundingBox<T>::shrink(Bound bnd, Axis axis, real_type position)
{
    real_type p = points_[to_int(bnd)][to_int(axis)];
    if (bnd == Bound::lo)
    {
        p = std::fmax(p, position);
    }
    else
    {
        p = std::fmin(p, position);
    }
    points_[to_int(bnd)][to_int(axis)] = p;
}

//---------------------------------------------------------------------------//
/*!
 * Increase (expand) the bounding box's extent along an axis.
 *
 * If the point is outside the box, the box is expanded so the given boundary
 * is on that point. Otherwise no change is made.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION void
BoundingBox<T>::grow(Bound bnd, Axis axis, real_type position)
{
    real_type p = points_[to_int(bnd)][to_int(axis)];
    if (bnd == Bound::lo)
    {
        p = std::fmin(p, position);
    }
    else
    {
        p = std::fmax(p, position);
    }
    points_[to_int(bnd)][to_int(axis)] = p;
}

//---------------------------------------------------------------------------//
/*!
 * Increase (expand) the bounding box's extent along an axis.
 *
 * If the point is outside the box, the box is expanded so the given boundary
 * is on that point. Otherwise no change is made.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION void
BoundingBox<T>::grow(Axis axis, real_type position)
{
    this->grow(Bound::lo, axis, position);
    this->grow(Bound::hi, axis, position);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
