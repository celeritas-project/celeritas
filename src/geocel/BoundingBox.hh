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

#if !CELER_DEVICE_COMPILE
#    include <ostream>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned bounding box.
 *
 * Bounding boxes "contain" all points inside \em and on their faces.
 * See \c is_inside in \c BoundingBoxUtils.hh .
 *
 * The default bounding box is "null", which has at least one \c lower
 * coordinate strictly greater than its \c upper coordinate: it evaluates to
 * \c false .
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
    using Extents = Array<real_type, 2>;
    using Extents3 = Array<Extents, 3>;
    //!@}

  public:
    // Construct from infinite extents
    static inline CELER_FUNCTION BoundingBox from_infinite() noexcept;

    // Construct from unchecked lower/upper bounds
    static CELER_CONSTEXPR_FUNCTION BoundingBox from_unchecked(
        Real3 const& lower, Real3 const& upper) noexcept;

    // Construct from unchecked lo/hi extents
    static CELER_CONSTEXPR_FUNCTION BoundingBox from_unchecked(
        Extents3 const&) noexcept;

    // Construct in unassigned state
    CELER_CONSTEXPR_FUNCTION BoundingBox() noexcept;

    // Construct from upper and lower points
    inline CELER_FUNCTION BoundingBox(
        Real3 const& lower, Real3 const& upper) noexcept(!CELERITAS_DEBUG);

    // Construct from lo/hi extents (transposed layout)
    inline CELER_FUNCTION
    BoundingBox(Extents3 const& extents) noexcept(!CELERITAS_DEBUG);

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
        noexcept(!CELERITAS_DEBUG)
    {
        CELER_EXPECT(b != Bound::size_);
        return points_[to_int(b)];
    }

    //! Access a bounding point coordinate (const ref to support LDG)
    CELER_CONSTEXPR_FUNCTION real_type const&
    point(Bound b, Axis ax) const& noexcept(!CELERITAS_DEBUG)
    {
        CELER_EXPECT(ax != Axis::size_);
        return points_[to_int(b)][to_int(ax)];
    }

    //! Access a bounding point coordinate
    CELER_CONSTEXPR_FUNCTION real_type point(
        Bound b, Axis ax) const&& noexcept(!CELERITAS_DEBUG)
    {
        return this->point(b, ax);
    }

    // Whether the bbox is non-null
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const noexcept;

    //// MUTATORS ////

    // Reduce the bounding box's extent along an axis
    CELER_CONSTEXPR_FUNCTION void shrink(Bound b, Axis ax, real_type p);

    // Increase the bounding box's extent along an axis
    CELER_CONSTEXPR_FUNCTION void grow(Bound b, Axis ax, real_type p);

    // Increase the bounding box's extent on both bounds
    CELER_CONSTEXPR_FUNCTION void grow(Axis ax, real_type p);

    //// FRIENDS ////

    //! Test equality of two bounding boxes
    CELER_CONSTEXPR_FUNCTION friend bool
    operator==(BoundingBox const& lhs, BoundingBox const& rhs)
    {
        return lhs.points_ == rhs.points_;
    }

    //! Test inequality of two bounding boxes
    CELER_CONSTEXPR_FUNCTION friend bool
    operator!=(BoundingBox const& lhs, BoundingBox const& rhs)
    {
        return !(lhs == rhs);
    }

    //! Allow loading via ldg
    CELER_CONSTEXPR_FUNCTION friend BoundingBox
    ldg(BoundingBox const* bb) noexcept
    {
        return BoundingBox{std::true_type{}, ldg(&bb->points_)};
    }

#if !CELER_DEVICE_COMPILE
    //! Write box to a stream
    friend std::ostream& operator<<(std::ostream& os, BoundingBox const& bbox)
    {
        os << '{';
        if (bbox)
        {
            os << bbox.lower() << ", " << bbox.upper();
        }
        os << '}';
        return os;
    }
#endif

  private:
    using Points = Array<Real3, 2>;

    Points points_;  //!< lo/hi points

    //// PRIVATE HELPERS ////

    // Construct internally without validation (using tag type)
    CELER_CONSTEXPR_FUNCTION
    BoundingBox(std::true_type, Points const& points) noexcept;
    CELER_CONSTEXPR_FUNCTION
    BoundingBox(std::true_type, Extents3 const& extents) noexcept;

    //! Set a bounding point coordinate (different signature to allow private)
    CELER_CONSTEXPR_FUNCTION void
    point(Bound b, Axis ax, real_type p) & noexcept
    {
        points_[to_int(b)][to_int(ax)] = p;
    }
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
 * Determine if a point is contained in a bounding box.
 *
 * No point is ever contained in a null bounding box. A degenerate bounding
 * box will return "true" for any point on its face.
 */
template<class T, class U>
CELER_CONSTEXPR_FUNCTION bool
is_inside(BoundingBox<T> const& bb, Array<U, 3> const& p) noexcept
{
    // clang-format off
    using B = Bound; using A = Axis;
    return    bb.point(B::lo, A::x) <= p[0] && p[0] <= bb.point(B::hi, A::x)
           && bb.point(B::lo, A::y) <= p[1] && p[1] <= bb.point(B::hi, A::y)
           && bb.point(B::lo, A::z) <= p[2] && p[2] <= bb.point(B::hi, A::z);
    // clang-format on
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create a bounding box with infinite extents.
 */
template<class T>
CELER_FUNCTION BoundingBox<T> BoundingBox<T>::from_infinite() noexcept
{
    constexpr real_type inf = numeric_limits<real_type>::infinity();
    return {{-inf, -inf, -inf}, {inf, inf, inf}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box from unchecked lower/upper points.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>
BoundingBox<T>::from_unchecked(Real3 const& lo, Real3 const& hi) noexcept
{
    return BoundingBox{std::true_type{}, Points{lo, hi}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box from unchecked lo/hi extents.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>
BoundingBox<T>::from_unchecked(Extents3 const& extents) noexcept
{
    return BoundingBox{std::true_type{}, extents};
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
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::BoundingBox() noexcept
{
    constexpr real_type inf = numeric_limits<real_type>::infinity();
    for (auto ax : {Axis::x, Axis::y, Axis::z})
    {
        this->point(Bound::lo, ax, inf);
        this->point(Bound::hi, ax, -inf);
    }
    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
/*!
 * Create a non-null bounding box from two points.
 *
 * The lower and upper points are allowed to be equal (an empty bounding box
 * at a single point) but upper must not be less than lower.
 */
template<class T>
CELER_FUNCTION BoundingBox<T>::BoundingBox(
    Real3 const& lo, Real3 const& hi) noexcept(!CELERITAS_DEBUG)
    : BoundingBox{std::true_type{}, Points{lo, hi}}
{
    CELER_EXPECT(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Create a non-null bounding box from lo/hi extents.
 */
template<class T>
CELER_FUNCTION
BoundingBox<T>::BoundingBox(Extents3 const& extents) noexcept(!CELERITAS_DEBUG)
    : BoundingBox{std::true_type{}, extents}
{
    CELER_EXPECT(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box from points without validation (internal).
 */
template<class T>
CELER_CONSTEXPR_FUNCTION
BoundingBox<T>::BoundingBox(std::true_type, Points const& points) noexcept
    : points_{points}
{
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box from extents without validation (internal).
 */
template<class T>
CELER_CONSTEXPR_FUNCTION
BoundingBox<T>::BoundingBox(std::true_type, Extents3 const& extents) noexcept
{
    for (auto ax : {Axis::x, Axis::y, Axis::z})
    {
        for (auto b : {Bound::lo, Bound::hi})
        {
            this->point(b, ax, extents[to_int(ax)][to_int(b)]);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox contains any point in space.
 *
 * A null box contains no points, and a degenerate point/edge/face contains
 * that point.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::operator bool() const noexcept
{
    // clang-format off
    return this->point(Bound::lo, Axis::x) <= this->point(Bound::hi, Axis::x)
        && this->point(Bound::lo, Axis::y) <= this->point(Bound::hi, Axis::y)
        && this->point(Bound::lo, Axis::z) <= this->point(Bound::hi, Axis::z);
    // clang-format on
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
BoundingBox<T>::shrink(Bound b, Axis ax, real_type v)
{
    real_type p = this->point(b, ax);
    if (b == Bound::lo)
    {
        p = std::fmax(p, v);
    }
    else
    {
        p = std::fmin(p, v);
    }
    this->point(b, ax, p);
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
BoundingBox<T>::grow(Bound b, Axis ax, real_type v)
{
    real_type p = this->point(b, ax);
    if (b == Bound::lo)
    {
        p = std::fmin(p, v);
    }
    else
    {
        p = std::fmax(p, v);
    }
    this->point(b, ax, p);
}

//---------------------------------------------------------------------------//
/*!
 * Increase (expand) the bounding box's extent along an axis.
 *
 * If the point is outside the box, the box is expanded so the given boundary
 * is on that point. Otherwise no change is made.
 *
 * \post If the box is non-null, some point on the axis \c is_inside the
 * bounding box.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION void BoundingBox<T>::grow(Axis ax, real_type p)
{
    this->grow(Bound::lo, ax, p);
    this->grow(Bound::hi, ax, p);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
