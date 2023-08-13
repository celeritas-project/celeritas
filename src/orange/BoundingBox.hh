//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBox.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned bounding box.
 *
 * The bounding box can be constructed in an unassigned state, in which lower
 * and upper cannot be called.
 */
template<class T>
class BoundingBox
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using Real3 = Array<real_type, 3>;
    //!@}

    // Construct from infinite extents
    static inline CELER_FUNCTION BoundingBox from_infinite();

    // Construct in unassigned state
    CELER_CONSTEXPR_FUNCTION BoundingBox();

    // Construct from upper and lower points
    inline CELER_FUNCTION BoundingBox(Real3 const& lower, Real3 const& upper);

    //// ACCESSORS ////

    //! Lower bbox coordinate
    CELER_CONSTEXPR_FUNCTION Real3 const& lower() const { return lower_; }

    //! Upper bbox coordinate
    CELER_CONSTEXPR_FUNCTION Real3 const& upper() const { return upper_; }

    // Whether the bbox is nondegenerate
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const;

  private:
    Real3 lower_;
    Real3 upper_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Bounding box for host metadata
using BBox = BoundingBox<::celeritas::real_type>;

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
 * Create a degenerate bounding box.
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
 * Create a valid bounding box from two points.
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
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox is in a nondegenerate state.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION BoundingBox<T>::operator bool() const
{
    return lower_[to_int(Axis::x)] <= upper_[to_int(Axis::x)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
