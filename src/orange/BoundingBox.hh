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
 * This is currently a pretty boring class; it will be extended as needed.
 * We'll add a BoundingBoxUtils class for all the fancy operations that need
 * only the accessors we provide.
 *
 * The bounding box can be constructed in an unassigned state, in which lower
 * and upper cannot be called.
 */
class BoundingBox
{
  public:
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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create a bounding box with infinite extents.
 */
CELER_FUNCTION BoundingBox BoundingBox::from_infinite()
{
    constexpr real_type inf = numeric_limits<real_type>::infinity();
    return {{-inf, -inf, -inf}, {inf, inf, inf}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box in an invalid state.
 */
CELER_FUNCTION BoundingBox::BoundingBox()
{
    lower_[0] = numeric_limits<real_type>::quiet_NaN();
}

//---------------------------------------------------------------------------//
/*!
 * Create a valid bounding box from two points.
 *
 * The lower and upper points are allowed to be equal (an empty bounding box
 * at a single point) but upper must not be less than lower.
 */
CELER_FUNCTION BoundingBox::BoundingBox(Real3 const& lo, Real3 const& hi)
    : lower_(lo), upper_(hi)
{
    CELER_EXPECT(lower_[0] <= upper_[0] && lower_[1] <= upper_[1]
                 && lower_[2] <= upper_[2]);
}

//---------------------------------------------------------------------------//
/*!
 * Lower bbox coordinate (must be valid).
 */
CELER_FUNCTION Real3 const& BoundingBox::lower() const
{
    CELER_EXPECT(*this);
    return lower_;
}

//---------------------------------------------------------------------------//
/*!
 * Upper bbox coordinate (must be valid).
 */
CELER_FUNCTION Real3 const& BoundingBox::upper() const
{
    CELER_EXPECT(*this);
    return upper_;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the bbox is in an assigned state.
 */
CELER_FUNCTION BoundingBox::operator bool() const
{
    return !std::isnan(lower_[0]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
