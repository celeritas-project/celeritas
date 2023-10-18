//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBox.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"

#include "OrangeTypes.hh"

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
 * because at least one lower coorindate is equal to the corresponding upper
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
    CELER_CONSTEXPR_FUNCTION Real3 const& lower() const { return lower_; }

    //! Upper bbox coordinate
    CELER_CONSTEXPR_FUNCTION Real3 const& upper() const { return upper_; }

    // Whether the bbox is non-null
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const;

    //// MUTATORS ////

    // Intersect in place with a half-space
    CELER_CONSTEXPR_FUNCTION void
    clip(Sense sense, Axis axis, real_type position);

  private:
    Real3 lower_;
    Real3 upper_;

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
    lower_ = {inf, inf, inf};
    upper_ = {-inf, -inf, -inf};
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
    : lower_(lo), upper_(hi)
{
    if constexpr (CELERITAS_DEBUG)
    {
        for (auto ax : {Axis::x, Axis::y, Axis::z})
        {
            CELER_EXPECT(lower_[to_int(ax)] <= upper_[to_int(ax)]);
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
    : lower_(lo), upper_(hi)
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox is in a valid state.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::operator bool() const
{
    return lower_[to_int(Axis::x)] <= upper_[to_int(Axis::x)]
           && lower_[to_int(Axis::y)] <= upper_[to_int(Axis::y)]
           && lower_[to_int(Axis::z)] <= upper_[to_int(Axis::z)];
}

//---------------------------------------------------------------------------//
/*!
 * Intersect in place with a half-space.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION void
BoundingBox<T>::clip(Sense sense, Axis axis, real_type position)
{
    if (sense == Sense::inside)
    {
        upper_[to_int(axis)] = ::celeritas::min(upper_[to_int(axis)], position);
    }
    else
    {
        lower_[to_int(axis)] = ::celeritas::max(lower_[to_int(axis)], position);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
