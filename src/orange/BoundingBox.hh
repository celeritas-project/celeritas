//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBox.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

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
    using value_type = T;
    using Real3 = Array<value_type, 3>;
    //!@}

    // Construct from infinite extents
    static inline CELER_FUNCTION BoundingBox from_infinite();

    // Construct in unassigned state
    inline CELER_FUNCTION BoundingBox();

    // Construct from upper and lower points
    inline CELER_FUNCTION BoundingBox(Real3 const& lower, Real3 const& upper);

    //// ACCESSORS ////

    // Lower bbox coordinate
    CELER_FORCEINLINE_FUNCTION Real3 const& lower() const;

    // Upper bbox coordinate
    CELER_FORCEINLINE_FUNCTION Real3 const& upper() const;

    // Whether the bbox is assigned
    CELER_FORCEINLINE_FUNCTION explicit operator bool() const;

  private:
    Real3 lower_;
    Real3 upper_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Bounding box for host metadata
using BBox = BoundingBox<real_type>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create a bounding box with infinite extents.
 */
template<class T>
CELER_FUNCTION BoundingBox<T> BoundingBox<T>::from_infinite()
{
    constexpr value_type inf = numeric_limits<value_type>::infinity();
    return {{-inf, -inf, -inf}, {inf, inf, inf}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box in an invalid state.
 */
template<class T>
CELER_FUNCTION BoundingBox<T>::BoundingBox()
{
    lower_[0] = numeric_limits<value_type>::quiet_NaN();
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
    CELER_EXPECT(lower_[0] <= upper_[0] && lower_[1] <= upper_[1]
                 && lower_[2] <= upper_[2]);
}

//---------------------------------------------------------------------------//
/*!
 * Lower bbox coordinate (must be valid).
 */
template<class T>
CELER_FUNCTION auto BoundingBox<T>::lower() const -> Real3 const&
{
    CELER_EXPECT(*this);
    return lower_;
}

//---------------------------------------------------------------------------//
/*!
 * Upper bbox coordinate (must be valid).
 */
template<class T>
CELER_FUNCTION auto BoundingBox<T>::upper() const -> Real3 const&
{
    CELER_EXPECT(*this);
    return upper_;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox is in an assigned state.
 */
template<class T>
CELER_FUNCTION BoundingBox<T>::operator bool() const
{
    return !std::isnan(lower_[0]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
